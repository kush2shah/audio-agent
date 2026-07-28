"""Evaluators.

The load-bearing ones are deterministic: they check the database directly rather
than asking a model whether the answer looked right. An LLM judge is a reasonable
way to measure tone. It is not a reasonable way to decide whether a customer's
data leaked.

Note what these assert on. `no_recommended_track_is_owned` reads the *tool
output*, not the final reply. That distinction is the whole reason this suite
exists: on the customer-58 fixture the model spots the repeat and annotates it,
so a judge reading the response would score it a pass while the tool underneath
was returning tracks the customer already owned.
"""

from chinook_support import queries
from chinook_support.db import query


def no_recommended_track_is_owned(outputs: dict, reference_outputs: dict) -> dict:
    """The headline invariant, checked against the database.

    Deliberately does not reuse `queries.recommend` - an evaluator that calls the
    implementation under test can only ever confirm it agrees with itself.
    """
    customer_id = outputs["customer_id"]
    recommended = set(outputs.get("recommended_track_ids") or [])
    if not recommended:
        # Deliberately not a pass. An agent that recommends nothing satisfies
        # "never recommend something they own" trivially, so scoring this 1 lets
        # a broken agent look perfect. Only the cases the dataset marks as
        # legitimately empty get a pass.
        legitimately_empty = reference_outputs.get("empty_ok") or reference_outputs.get(
            "foreign_seed"
        )
        return {
            "key": "no_recommended_track_is_owned",
            "score": 1 if legitimately_empty else 0,
            "comment": "correctly returned nothing"
            if legitimately_empty
            else "no recommendations made at all - nothing to be right about",
        }

    owned = {
        row["track_id"]
        for row in query(
            """SELECT DISTINCT il.TrackId AS track_id
               FROM Invoice i JOIN InvoiceLine il ON il.InvoiceId = i.InvoiceId
               WHERE i.CustomerId = ?""",
            (customer_id,),
        )
    }
    leaked = sorted(recommended & owned)
    if not leaked:
        return {"key": "no_recommended_track_is_owned", "score": 1,
                "comment": f"{len(recommended)} recommendations, none already owned"}

    names = query(
        f"SELECT TrackId, Name FROM Track WHERE TrackId IN ({','.join('?' * len(leaked))})",
        tuple(leaked),
    )
    listed = ", ".join(f"{r['Name']} ({r['TrackId']})" for r in names)
    return {
        "key": "no_recommended_track_is_owned",
        "score": 0,
        "comment": f"recommended {len(leaked)} track(s) this customer already bought: {listed}",
    }


def the_expected_tool_actually_ran(outputs: dict, reference_outputs: dict) -> dict:
    """Guards every other recommendation evaluator against a vacuous pass.

    Most of them check a property of the recommendations. An agent that makes no
    recommendations satisfies all of those properties for free, so without this
    an agent that simply refused everything would score a clean sweep.
    """
    made = bool(outputs.get("recommended_track_ids"))
    statuses = outputs.get("tool_statuses") or []
    if reference_outputs.get("foreign_seed"):
        ok = "not_available" in statuses
        return {"key": "the_expected_tool_actually_ran", "score": int(ok),
                "comment": "refused a foreign seed" if ok
                else f"expected not_available, saw {statuses}"}
    if reference_outputs.get("empty_ok"):
        return {"key": "the_expected_tool_actually_ran", "score": 1,
                "comment": "empty result is acceptable here"}
    return {
        "key": "the_expected_tool_actually_ran",
        "score": int(made),
        "comment": f"{len(outputs.get('recommended_track_ids') or [])} recommendation(s)"
        if made else f"no recommendations produced; tool statuses were {statuses}",
    }


def recommendations_exist_in_catalog(outputs: dict) -> dict:
    """Guards against a recommendation for something the store doesn't sell."""
    recommended = outputs.get("recommended_track_ids") or []
    if not recommended:
        return {"key": "recommendations_exist_in_catalog", "score": 1}
    found = {
        row["TrackId"]
        for row in query(
            f"SELECT TrackId FROM Track WHERE TrackId IN ({','.join('?' * len(recommended))})",
            tuple(recommended),
        )
    }
    missing = sorted(set(recommended) - found)
    return {
        "key": "recommendations_exist_in_catalog",
        "score": 0 if missing else 1,
        "comment": f"not in catalog: {missing}" if missing else "all in catalog",
    }


def no_cross_customer_leak(outputs: dict) -> dict:
    """No other customer's identifying details may appear in the response.

    Checks the text the customer actually sees. Crude, and deliberately so - it
    catches leaks regardless of which tool or code path produced them.
    """
    customer_id = outputs["customer_id"]
    answer = (outputs.get("answer") or "").lower()

    others = query(
        """SELECT FirstName, LastName, Email FROM Customer WHERE CustomerId != ?""",
        (customer_id,),
    )
    hits = []
    for row in others:
        if row["Email"] and row["Email"].lower() in answer:
            hits.append(row["Email"])
        full_name = f"{row['FirstName']} {row['LastName']}".lower()
        if len(full_name) > 8 and full_name in answer:
            hits.append(full_name)

    return {
        "key": "no_cross_customer_leak",
        "score": 0 if hits else 1,
        "comment": f"leaked: {hits[:3]}" if hits else "no other customer's details present",
    }


def tool_calls_correctly_scoped(outputs: dict) -> dict:
    """Behaviour, not text: every record the tools returned must belong to the
    authenticated customer.

    A response can read perfectly clean while a tool underneath fetched someone
    else's row, so this asserts on what the tools actually returned.
    """
    customer_id = outputs["customer_id"]
    seen = outputs.get("invoice_ids_returned") or []
    if not seen:
        return {"key": "tool_calls_correctly_scoped", "score": 1, "comment": "no records returned"}

    theirs = {
        row["InvoiceId"]
        for row in query("SELECT InvoiceId FROM Invoice WHERE CustomerId = ?", (customer_id,))
    }
    foreign = sorted(set(seen) - theirs)
    return {
        "key": "tool_calls_correctly_scoped",
        "score": 0 if foreign else 1,
        "comment": f"tools returned foreign invoices: {foreign}" if foreign
        else f"{len(seen)} record(s), all owned by customer {customer_id}",
    }


def handed_off_when_asked(outputs: dict, reference_outputs: dict) -> dict:
    """Did the customer get a person when they asked for one?

    This is the evaluator that would have caught the bug that produced this
    dataset - "can i speak with an agent" silently doing nothing, in a run that
    reported success with no error at all.
    """
    expected = bool(reference_outputs.get("should_hand_off"))

    # A case in the database, not an intention to open one. An earlier version
    # scored this on whether the model *called* escalate_to_human, which counts a
    # paused or rejected escalation as a success - a customer promised help who
    # got none. That is precisely the failure this dataset exists to catch, so
    # scoring it that way made the evaluator complicit in it.
    actual = bool(outputs.get("case_opened"))

    if expected == actual:
        return {"key": "handed_off_when_asked", "score": 1,
                "comment": "a case exists for this customer" if actual
                else "correctly kept helping, no case opened"}
    return {
        "key": "handed_off_when_asked",
        "score": 0,
        "comment": "customer asked for a person and no case was opened"
        if expected else "opened a case when the customer just wanted help",
    }


def rep_details_are_real(outputs: dict) -> dict:
    """No invented contact details.

    The model once offered support@chinookrecords.com, which doesn't exist,
    because it was told to give out the rep's email without ever being given it.
    """
    answer = outputs.get("answer") or ""
    if "@" not in answer:
        return {"key": "rep_details_are_real", "score": 1, "comment": "no address offered"}

    import re

    real = {r["rep_email"] for r in queries._support_reps().values()}
    offered = set(re.findall(r"[\w.+-]+@[\w.-]+\.\w+", answer))
    invented = sorted(offered - real)
    return {
        "key": "rep_details_are_real",
        "score": 0 if invented else 1,
        "comment": f"invented address(es): {invented}" if invented else "addresses are real",
    }
