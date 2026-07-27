"""The back-office case investigator - a Deep Agent, deliberately off the chat path.

When the support agent hands a customer to a person, it opens a case. This works
that queue: for each open case it pulls the customer's history, works out who they
are and what they like, and writes a brief the rep can read in thirty seconds
before replying.

Why this is a Deep Agent and the support bot is not:

    support bot     one customer, one question, seconds, a fixed small set of
                    tools. `create_agent()` - a model/tool loop is the whole job.

    investigator    an open-ended queue, many steps per case, needs to plan,
                    accumulate findings, and write them somewhere.
                    `create_deep_agent()` - planning and a filesystem are the job.

The boundary is the point. A customer asking "what did I buy" must never trigger
a planning loop, and a rep preparing for a difficult conversation shouldn't be
limited to what fits in one turn.

Note what it does *not* do. Chinook has no chargebacks, disputes, or payment
records - I checked, and there isn't a single customer who bought the same track
twice. So this doesn't investigate billing errors, because there are none to
find. It does what the data supports: prepares a brief from real purchase
history.

The tools here are staff-facing and take an explicit customer_id, unlike the
customer-facing tools where identity is injected and unforgeable. Different
principal, different contract - a rep is *supposed* to be able to look up the
customer whose case they're working.
"""

from langchain.tools import tool

from . import cases
from .db import query


@tool
def list_open_cases() -> list[dict]:
    """List support cases waiting for a rep."""
    with cases._connect() as con:
        rows = con.execute(
            "SELECT case_id, customer_id, rep_id, reason, opened_at "
            "FROM support_case WHERE status = 'open' ORDER BY opened_at"
        ).fetchall()
        return [dict(row) for row in rows]


@tool
def customer_profile(customer_id: int) -> dict:
    """Who this customer is: how long, how much, and what they listen to.

    Args:
        customer_id: The customer on the case being worked.
    """
    totals = query(
        """SELECT COUNT(*) AS orders,
                  ROUND(SUM(Total), 2) AS lifetime_spend,
                  MIN(DATE(InvoiceDate)) AS first_order,
                  MAX(DATE(InvoiceDate)) AS last_order
           FROM Invoice WHERE CustomerId = ?""",
        (customer_id,),
    )
    genres = query(
        """SELECT g.Name AS genre, COUNT(*) AS tracks
           FROM Invoice i
           JOIN InvoiceLine il ON il.InvoiceId = i.InvoiceId
           JOIN Track t ON t.TrackId = il.TrackId
           LEFT JOIN Genre g ON g.GenreId = t.GenreId
           WHERE i.CustomerId = ?
           GROUP BY g.Name ORDER BY tracks DESC LIMIT 5""",
        (customer_id,),
    )
    artists = query(
        """SELECT ar.Name AS artist, COUNT(*) AS tracks
           FROM Invoice i
           JOIN InvoiceLine il ON il.InvoiceId = i.InvoiceId
           JOIN Track t ON t.TrackId = il.TrackId
           JOIN Album al ON al.AlbumId = t.AlbumId
           JOIN Artist ar ON ar.ArtistId = al.ArtistId
           WHERE i.CustomerId = ?
           GROUP BY ar.Name ORDER BY tracks DESC LIMIT 5""",
        (customer_id,),
    )
    return {**(totals[0] if totals else {}), "top_genres": genres, "top_artists": artists}


@tool
def order_history(customer_id: int, limit: int = 10) -> list[dict]:
    """This customer's orders, newest first.

    Args:
        customer_id: The customer on the case.
        limit: How many orders to return.
    """
    return query(
        """SELECT i.InvoiceId AS invoice_id, DATE(i.InvoiceDate) AS date,
                  i.Total AS total, COUNT(il.InvoiceLineId) AS items
           FROM Invoice i JOIN InvoiceLine il ON il.InvoiceId = i.InvoiceId
           WHERE i.CustomerId = ?
           GROUP BY i.InvoiceId ORDER BY i.InvoiceDate DESC LIMIT ?""",
        (customer_id, limit),
    )


@tool
def unsold_catalog_for_artist(artist_name: str, customer_id: int) -> list[dict]:
    """Albums we stock by an artist, with how much of each this customer owns.

    Reports per album: total tracks, how many they already have, how many are
    new. Returning bare unowned tracks was not enough - the agent read a list of
    unowned tracks and wrote "this album is fully unowned" about a record the
    customer already had one track from. The tool was right and the summary
    wasn't, so the tool now carries the fact the summary needs.

    Args:
        artist_name: Artist to check.
        customer_id: Customer whose purchases to account for.
    """
    return query(
        """SELECT al.Title AS album,
                  COUNT(*) AS tracks_total,
                  SUM(CASE WHEN owned.TrackId IS NULL THEN 0 ELSE 1 END) AS tracks_owned,
                  SUM(CASE WHEN owned.TrackId IS NULL THEN 1 ELSE 0 END) AS tracks_new
           FROM Track t
           JOIN Album al ON al.AlbumId = t.AlbumId
           JOIN Artist ar ON ar.ArtistId = al.ArtistId
           LEFT JOIN (
               SELECT DISTINCT il.TrackId
               FROM Invoice i JOIN InvoiceLine il ON il.InvoiceId = i.InvoiceId
               WHERE i.CustomerId = ?
           ) owned ON owned.TrackId = t.TrackId
           WHERE ar.Name = ?
           GROUP BY al.AlbumId
           HAVING tracks_new > 0
           ORDER BY tracks_owned DESC, tracks_new DESC
           LIMIT 8""",
        (customer_id, artist_name),
    )


INVESTIGATOR_PROMPT = """You prepare case briefs for Chinook Records support reps.

For each open case: look up who the customer is, what they've bought, and what \
they listen to. Then write a brief the rep can read in thirty seconds before \
replying.

A good brief says who this customer is to us (how long, how much, what they \
love), what the case is about, and one concrete thing the rep could offer - \
something we stock by an artist they already like and haven't bought yet.

Save each brief to `briefs/case-<id>.md`. Be direct; reps are busy."""


def build_investigator():
    """Assembled lazily - importing deepagents pulls in a lot, and the chat agent
    never needs it."""
    from deepagents import create_deep_agent

    return create_deep_agent(
        model="anthropic:claude-sonnet-4-6",
        tools=[list_open_cases, customer_profile, order_history, unsold_catalog_for_artist],
        system_prompt=INVESTIGATOR_PROMPT,
    )
