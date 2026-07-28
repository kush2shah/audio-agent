"""The back-office case investigator - a Deep Agent, deliberately off the chat path.

When the support agent hands a customer to a person it opens a case. This works
that queue, and its job is diagnosis, not summary: *why* did this person end up
escalating?

That distinction is what makes it Deep Agent work. Summarising a customer is a
fixed pipeline - fetch, format, done - and `create_agent()` handles it fine.
Diagnosis branches: if they're lapsed you look at what changed before they went
quiet; if their spend collapsed you look at what they stopped buying; if our
recommender was feeding them things they already owned you check how far back it
goes. What you look at second depends on what you found first, which is exactly
what planning and a scratchpad are for.

    support bot     one customer, one question, seconds, fixed tools.
                    create_agent() - a model/tool loop is the whole job.

    investigator    an open-ended question, branching evidence, minutes,
                    writes findings down. create_deep_agent().

A customer asking "what did I buy" must never trigger a planning loop.

Note what this deliberately doesn't do. Chinook has no chargebacks, disputes, or
payment records - not one customer bought the same track twice - so it doesn't
investigate billing errors, because there are none. It investigates what the data
can actually answer.

The tools here are staff-facing and take an explicit customer_id, unlike the
customer-facing tools where identity is injected and unforgeable. Different
principal, different contract: a rep is *supposed* to be able to look up the
customer whose case they're working. That also means this agent must never be
reachable by customer traffic - a deployment boundary, not a code one.
"""

from langchain.tools import tool

from . import cases, queries
from .db import query

# The dataset's "today". Chinook's last invoice is 2025-12-22, so real-world now
# would make every customer look lapsed.
TODAY = "2025-12-22"


@tool
def open_cases() -> list[dict]:
    """Support cases waiting for a rep, oldest first."""
    with cases._connect() as con:
        rows = con.execute(
            "SELECT case_id, customer_id, rep_id, reason, opened_at "
            "FROM support_case WHERE status = 'open' ORDER BY opened_at"
        ).fetchall()
        return [dict(row) for row in rows]


@tool
def purchase_timeline(customer_id: int) -> dict:
    """Order history over time: is this customer growing, shrinking, or gone?

    Reports every order, how long they've been quiet, and whether their recent
    spend is below their early spend. Use it to decide whether this is a lapsed
    customer, a declining one, or someone who was fine until something specific
    happened.

    Args:
        customer_id: The customer on the case.
    """
    orders = query(
        """SELECT InvoiceId AS invoice_id, DATE(InvoiceDate) AS date, Total AS total
           FROM Invoice WHERE CustomerId = ? ORDER BY InvoiceDate""",
        (customer_id,),
    )
    if not orders:
        return {"orders": [], "note": "no purchase history"}

    half = len(orders) // 2 or 1
    early = sum(o["total"] for o in orders[:half]) / half
    late_orders = orders[half:] or orders[-1:]
    late = sum(o["total"] for o in late_orders) / len(late_orders)
    quiet = query(
        "SELECT CAST(julianday(?) - julianday(MAX(InvoiceDate)) AS INT) AS days "
        "FROM Invoice WHERE CustomerId = ?",
        (TODAY, customer_id),
    )[0]["days"]

    return {
        "orders": orders,
        "order_count": len(orders),
        "days_since_last_order": quiet,
        "avg_order_early": round(early, 2),
        "avg_order_recent": round(late, 2),
        "spend_direction": "declining" if late < early * 0.8
        else "growing" if late > early * 1.2 else "steady",
    }


@tool
def recommendation_audit(customer_id: int) -> dict:
    """Check whether our recommender has been suggesting things they already own.

    Replays the recommendation each of this customer's orders would have produced
    and counts how many suggestions were tracks they'd already bought. A customer
    repeatedly told to buy what they own has a concrete reason to disengage.

    Args:
        customer_id: The customer on the case.
    """
    owned = queries.owned_track_ids(customer_id)
    invoices = query(
        "SELECT InvoiceId AS invoice_id FROM Invoice WHERE CustomerId = ? ORDER BY InvoiceDate",
        (customer_id,),
    )

    findings = []
    for row in invoices:
        seed = queries.invoice_seed_profile(row["invoice_id"])
        if not seed["artist_ids"]:
            continue
        suggested = queries.recommend(customer_id, seed, None, None, 5, exclude_owned=False)
        repeats = [s["track"] for s in suggested if s["track_id"] in owned]
        if repeats:
            findings.append(
                {"seed_invoice": row["invoice_id"], "already_owned_suggested": repeats}
            )

    total = sum(len(f["already_owned_suggested"]) for f in findings)
    return {
        "orders_checked": len(invoices),
        "orders_with_bad_suggestions": len(findings),
        "wasted_suggestions": total,
        "detail": findings[:5],
    }


@tool
def catalog_gaps(customer_id: int) -> list[dict]:
    """Albums by artists this customer already likes, and how much of each they own.

    Gives a rep something concrete to offer. Counts are per album - total tracks,
    how many they own, how many are new - so nothing has to be inferred.

    Args:
        customer_id: The customer on the case.
    """
    return query(
        """SELECT ar.Name AS artist, al.Title AS album,
                  COUNT(*) AS tracks_total,
                  SUM(CASE WHEN owned.TrackId IS NULL THEN 0 ELSE 1 END) AS tracks_owned,
                  SUM(CASE WHEN owned.TrackId IS NULL THEN 1 ELSE 0 END) AS tracks_new
           FROM Track t
           JOIN Album al ON al.AlbumId = t.AlbumId
           JOIN Artist ar ON ar.ArtistId = al.ArtistId
           LEFT JOIN (SELECT DISTINCT il.TrackId FROM Invoice i
                      JOIN InvoiceLine il ON il.InvoiceId = i.InvoiceId
                      WHERE i.CustomerId = ?) owned ON owned.TrackId = t.TrackId
           WHERE ar.ArtistId IN (
               SELECT ar2.ArtistId
               FROM Invoice i JOIN InvoiceLine il ON il.InvoiceId = i.InvoiceId
               JOIN Track t2 ON t2.TrackId = il.TrackId
               JOIN Album al2 ON al2.AlbumId = t2.AlbumId
               JOIN Artist ar2 ON ar2.ArtistId = al2.ArtistId
               WHERE i.CustomerId = ?)
           GROUP BY al.AlbumId
           HAVING tracks_new > 0
           ORDER BY tracks_owned DESC, tracks_new DESC
           LIMIT 10""",
        (customer_id, customer_id),
    )


INVESTIGATOR_PROMPT = f"""You investigate escalated support cases for Chinook \
Records and write up what you find for the rep who has to reply.

Your job is diagnosis, not summary. The rep can read the order history themselves. \
What they need from you is *why this customer ended up escalating*, and what to do \
about it.

Work each case like this:

1. Look at the case, then form a hypothesis about what went wrong.
2. Test it with the tools. Follow what you find - if they went quiet, look at what \
changed before they went quiet; if our recommender was wasting their time, find out \
how long that had been happening.
3. Write findings to `briefs/case-<id>.md`.

Things worth checking, though not every one matters for every case: whether they've \
gone quiet (today's date is {TODAY}), whether their spend is falling, and whether \
our recommender has been suggesting tracks they already owned.

Every number you write must come from a tool result. If the evidence is thin, say \
the evidence is thin - a rep acting on a confident wrong guess is worse off than one \
told we don't know.

Keep briefs short. Lead with the likely cause, then the evidence, then one concrete \
thing the rep can offer."""


def build_investigator():
    """Assembled lazily - importing deepagents pulls in a lot, and the chat agent
    never needs it."""
    from deepagents import create_deep_agent

    return create_deep_agent(
        model="anthropic:claude-sonnet-4-6",
        tools=[open_cases, purchase_timeline, recommendation_audit, catalog_gaps],
        system_prompt=INVESTIGATOR_PROMPT,
    )


# Exported for langgraph.json so both agents can be opened side by side in
# Studio. The argument is visual: this one plans, branches and writes files; the
# support agent answers and stops.
graph = build_investigator()
