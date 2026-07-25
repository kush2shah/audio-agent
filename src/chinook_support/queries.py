"""Every SQL statement in the application, in one file.

Deliberately plain Python: no LangChain, no agent concepts. These functions take
an explicit `customer_id` and return rows.

Keeping this layer separate buys three things:

1. You can read the entire data-access surface of the agent in one sitting.
2. It is unit-testable without constructing an agent or spending a token.
3. The tool wrapper stays identical while the query underneath changes - which is
   how we swap the broken recommendation for the fixed one in the demo without
   touching the prompt, the model, or the tool signature.

Every customer-scoped query carries `AND CustomerId = ?`. That is the boundary.
"""

from .db import query, query_one

# --- Account ---------------------------------------------------------------


def invoices_for_customer(customer_id: int, limit: int) -> list[dict]:
    return query(
        """SELECT i.InvoiceId AS invoice_id,
                  DATE(i.InvoiceDate) AS date,
                  i.Total AS total,
                  COUNT(il.InvoiceLineId) AS item_count
           FROM Invoice i
           JOIN InvoiceLine il ON il.InvoiceId = i.InvoiceId
           WHERE i.CustomerId = ?
           GROUP BY i.InvoiceId
           ORDER BY i.InvoiceDate DESC
           LIMIT ?""",
        (customer_id, limit),
    )


def invoice_header(customer_id: int, invoice_id: int) -> dict | None:
    """Returns None both when the invoice doesn't exist and when it isn't theirs.

    The caller cannot tell those cases apart, and neither can the customer.
    """
    return query_one(
        """SELECT i.InvoiceId AS invoice_id,
                  DATE(i.InvoiceDate) AS date,
                  i.Total AS total
           FROM Invoice i
           WHERE i.InvoiceId = ? AND i.CustomerId = ?""",
        (invoice_id, customer_id),
    )


def invoice_lines(invoice_id: int) -> list[dict]:
    """Only ever called after invoice_header() has confirmed ownership."""
    return query(
        """SELECT t.TrackId AS track_id,
                  t.Name AS track,
                  ar.Name AS artist,
                  al.Title AS album,
                  COALESCE(g.Name, 'Unknown') AS genre,
                  il.UnitPrice AS unit_price,
                  il.Quantity AS quantity
           FROM InvoiceLine il
           JOIN Track t ON t.TrackId = il.TrackId
           LEFT JOIN Album al ON al.AlbumId = t.AlbumId
           LEFT JOIN Artist ar ON ar.ArtistId = al.ArtistId
           LEFT JOIN Genre g ON g.GenreId = t.GenreId
           WHERE il.InvoiceId = ?
           ORDER BY t.TrackId""",
        (invoice_id,),
    )


def owned_track_ids(customer_id: int) -> set[int]:
    """Every track this customer has ever bought."""
    rows = query(
        """SELECT DISTINCT il.TrackId AS track_id
           FROM Invoice i
           JOIN InvoiceLine il ON il.InvoiceId = i.InvoiceId
           WHERE i.CustomerId = ?""",
        (customer_id,),
    )
    return {r["track_id"] for r in rows}


def support_rep_for(customer_id: int) -> dict | None:
    """The employee already assigned to this customer in Chinook.

    Every customer has one. A handoff goes to *their* rep, not a generic queue -
    which is the difference between "someone will get back to you" and "Jane, who
    has handled your account since 2021, will."
    """
    return query_one(
        """SELECT e.EmployeeId AS rep_id,
                  e.FirstName || ' ' || e.LastName AS rep_name,
                  e.Title AS rep_title,
                  e.Email AS rep_email
           FROM Customer c
           JOIN Employee e ON e.EmployeeId = c.SupportRepId
           WHERE c.CustomerId = ?""",
        (customer_id,),
    )


# --- Catalog ---------------------------------------------------------------

# Shared by search and recommendations so every catalog row has the same shape.
_TRACK_SELECT = """
    SELECT t.TrackId AS track_id,
           t.Name AS track,
           ar.Name AS artist,
           al.Title AS album,
           COALESCE(g.Name, 'Unknown') AS genre,
           t.UnitPrice AS unit_price
    FROM Track t
    LEFT JOIN Album al ON al.AlbumId = t.AlbumId
    LEFT JOIN Artist ar ON ar.ArtistId = al.ArtistId
    LEFT JOIN Genre g ON g.GenreId = t.GenreId
"""


def search_tracks(term: str, genre: str | None, limit: int) -> list[dict]:
    """Match a term against track, artist, or album name.

    LIKE with bound parameters - the term is never concatenated into the SQL.
    """
    like = f"%{term}%"
    return query(
        _TRACK_SELECT
        + """WHERE (t.Name LIKE ? OR ar.Name LIKE ? OR al.Title LIKE ?)
               AND (? IS NULL OR g.Name = ?)
             ORDER BY
               CASE WHEN LOWER(t.Name) = LOWER(?) THEN 0 ELSE 1 END,
               t.TrackId
             LIMIT ?""",
        (like, like, like, genre, genre, term, limit),
    )


def exact_title_artists(term: str) -> list[dict]:
    """Distinct artists having a track by exactly this name.

    Used to detect genuine ambiguity: 'Believe' is three different songs by
    three different artists, and picking one silently is a guess the customer
    never asked us to make.
    """
    return query(
        """SELECT ar.ArtistId AS artist_id, ar.Name AS artist, t.TrackId AS track_id
           FROM Track t
           JOIN Album al ON al.AlbumId = t.AlbumId
           JOIN Artist ar ON ar.ArtistId = al.ArtistId
           WHERE LOWER(t.Name) = LOWER(?)
           ORDER BY ar.Name""",
        (term,),
    )


def invoice_seed_profile(invoice_id: int) -> dict:
    """The artists and genres on one invoice - what 'more like this' means."""
    rows = query(
        """SELECT DISTINCT ar.ArtistId AS artist_id, g.GenreId AS genre_id
           FROM InvoiceLine il
           JOIN Track t ON t.TrackId = il.TrackId
           LEFT JOIN Album al ON al.AlbumId = t.AlbumId
           LEFT JOIN Artist ar ON ar.ArtistId = al.ArtistId
           LEFT JOIN Genre g ON g.GenreId = t.GenreId
           WHERE il.InvoiceId = ?""",
        (invoice_id,),
    )
    return {
        "artist_ids": sorted({r["artist_id"] for r in rows if r["artist_id"]}),
        "genre_ids": sorted({r["genre_id"] for r in rows if r["genre_id"]}),
    }


# --- Recommendations -------------------------------------------------------
#
# Two versions of one query. They are identical except for ONE clause, which is
# the entire point of the demo - see EXCLUDE_OWNED below.
#
# Both rank candidates by how often the track has sold across the whole store.
# That is a completely reasonable thing to build: "customers who liked this
# artist also bought these" is what a real merchandising team would ship.

_RECOMMEND = """
    SELECT t.TrackId AS track_id,
           t.Name AS track,
           ar.Name AS artist,
           al.Title AS album,
           COALESCE(g.Name, 'Unknown') AS genre,
           t.UnitPrice AS unit_price,
           COUNT(il.InvoiceLineId) AS store_sales,
           CASE WHEN ar.ArtistId IN (%(artists)s) THEN 1 ELSE 0 END AS same_artist,
           CASE WHEN g.GenreId IN (%(genres)s) THEN 1 ELSE 0 END AS same_genre
    FROM Track t
    LEFT JOIN Album al ON al.AlbumId = t.AlbumId
    LEFT JOIN Artist ar ON ar.ArtistId = al.ArtistId
    LEFT JOIN Genre g ON g.GenreId = t.GenreId
    LEFT JOIN InvoiceLine il ON il.TrackId = t.TrackId
    WHERE (ar.ArtistId IN (%(artists)s) OR g.GenreId IN (%(genres)s))
      AND (? IS NULL OR g.Name = ?)
      AND (? IS NULL OR t.UnitPrice <= ?)
      %(exclude_owned)s
    GROUP BY t.TrackId
    ORDER BY same_artist DESC, store_sales DESC, t.TrackId
    LIMIT ?
"""

# The clause that separates v1 from v2. This is the whole bug, and the whole fix.
EXCLUDE_OWNED = """
      AND NOT EXISTS (
          SELECT 1
          FROM Invoice owner_inv
          JOIN InvoiceLine owned ON owned.InvoiceId = owner_inv.InvoiceId
          WHERE owner_inv.CustomerId = ?
            AND owned.TrackId = t.TrackId
      )
"""


def recommend(
    customer_id: int,
    seed: dict,
    genre: str | None,
    max_price: float | None,
    limit: int,
    exclude_owned: bool,
) -> list[dict]:
    """Candidate tracks for a customer, ranked by store-wide sales.

    `exclude_owned=False` is the v1 contract shipped in the demo: it never joins
    back to this customer's purchases, so it happily recommends tracks they
    already bought. Nothing is "forgotten" - the question is never asked.
    """
    artists = seed["artist_ids"] or [-1]
    genres = seed["genre_ids"] or [-1]

    sql = _RECOMMEND % {
        # Placeholder counts come from list lengths we control, never from input.
        "artists": ",".join("?" * len(artists)),
        "genres": ",".join("?" * len(genres)),
        "exclude_owned": EXCLUDE_OWNED if exclude_owned else "",
    }

    params: list = [*artists, *genres, *artists, *genres, genre, genre, max_price, max_price]
    if exclude_owned:
        params.append(customer_id)
    params.append(limit)

    return query(sql, tuple(params))
