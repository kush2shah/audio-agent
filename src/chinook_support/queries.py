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
