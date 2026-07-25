"""The staged failure, pinned down.

These tests exist so the demo can't quietly stop working. If v1 ever stops
failing, or v2 ever starts leaking, the suite goes red before the audience does.

Nothing here runs a model. The bug lives entirely below the model layer, which is
the point being made on stage.
"""

import pytest

from chinook_support import queries as q
from chinook_support.db import query
from chinook_support.tools.catalog import make_recommend_tracks, search_catalog

CUSTOMER, SEED_INVOICE = 58, 338


@pytest.fixture(scope="module")
def seed():
    return q.invoice_seed_profile(SEED_INVOICE)


@pytest.fixture(scope="module")
def owned():
    return q.owned_track_ids(CUSTOMER)


def test_v1_recommends_tracks_the_customer_already_owns(seed, owned):
    """The headline demo failure. If this passes, the demo has a story."""
    got = {r["track_id"] for r in q.recommend(CUSTOMER, seed, "Jazz", None, 5, False)}
    assert got & owned, "v1 no longer reproduces the bug - the demo needs a new failure"


def test_v1_returns_exactly_the_expected_owned_tracks(seed, owned):
    """Pin the specific tracks, so the demo script stays accurate.

    635 'Lemon Drop' and 639 "Don't Take Your Love From Me" are both on invoice
    338 - the very order used as the seed.
    """
    ids = [r["track_id"] for r in q.recommend(CUSTOMER, seed, "Jazz", None, 5, False)]
    assert ids == [625, 626, 635, 639, 627]
    assert set(ids) & owned == {635, 639}


def test_v1_is_deterministic(seed):
    """A demo that depends on a model misbehaving on cue is not a demo."""
    runs = {
        tuple(r["track_id"] for r in q.recommend(CUSTOMER, seed, "Jazz", None, 5, False))
        for _ in range(5)
    }
    assert len(runs) == 1


def test_v2_never_recommends_an_owned_track(seed, owned):
    got = {r["track_id"] for r in q.recommend(CUSTOMER, seed, "Jazz", None, 5, True)}
    assert not got & owned


def test_v2_still_returns_useful_results(seed):
    """Fixing the bug must not empty the shelf."""
    rows = q.recommend(CUSTOMER, seed, "Jazz", None, 5, True)
    assert len(rows) == 5
    assert all(r["genre"] == "Jazz" for r in rows)


def test_the_bug_is_widespread_not_a_lucky_fixture():
    """v1 leaks on a majority of invoices; v2 leaks on none.

    This is the number worth saying out loud: more than half of all
    recommendation requests would hand back something already purchased.
    """
    invoices = query("SELECT InvoiceId AS i, CustomerId AS c FROM Invoice")
    v1_leaks = v2_leaks = checked = 0

    for row in invoices:
        profile = q.invoice_seed_profile(row["i"])
        if not profile["artist_ids"]:
            continue
        owned_here = q.owned_track_ids(row["c"])
        checked += 1
        if any(r["track_id"] in owned_here for r in q.recommend(row["c"], profile, None, None, 5, False)):
            v1_leaks += 1
        if any(r["track_id"] in owned_here for r in q.recommend(row["c"], profile, None, None, 5, True)):
            v2_leaks += 1

    assert v2_leaks == 0
    assert v1_leaks / checked > 0.5, f"v1 leaked on only {v1_leaks}/{checked}"


def test_the_model_cannot_catch_the_bug_when_evidence_is_out_of_context():
    """The second demo fixture, and the one that actually matters.

    Customer 58's case is partly self-correcting: the seed invoice is in the
    conversation, so the model can notice a repeat and annotate it.

    Customer 35 is the honest version of the failure. Seeding from order 149
    (Iron Maiden) recommends three tracks they bought on order 355 - an order
    that appears nowhere in the conversation. The model has no evidence to
    cross-check against, so it presents all three with total confidence.

    Same bug in both cases. The only difference is whether the model got lucky.
    """
    seed = q.invoice_seed_profile(149)
    owned = q.owned_track_ids(35)
    on_seed = {line["track_id"] for line in q.invoice_lines(149)}

    recs = [r["track_id"] for r in q.recommend(35, seed, None, None, 5, False)]
    invisible = [t for t in recs if t in owned and t not in on_seed]

    assert invisible == [1208, 1226, 1244], (
        "the out-of-context leak fixture has drifted - the demo script names these"
    )
    # And v2 closes it.
    fixed = [r["track_id"] for r in q.recommend(35, seed, None, None, 5, True)]
    assert not set(fixed) & owned


def test_price_and_genre_constraints_are_enforced_by_the_tool(seed):
    rows = q.recommend(CUSTOMER, seed, "Jazz", 0.99, 5, True)
    assert all(r["unit_price"] <= 0.99 and r["genre"] == "Jazz" for r in rows)


def test_seed_invoice_ownership_is_checked(seed):
    """You can't seed recommendations off someone else's order."""
    tool = make_recommend_tracks(exclude_owned=True)
    assert "customer_id" not in tool.tool_call_schema.model_json_schema()["properties"]
    # Invoice 1 belongs to customer 2, so it must not be a valid seed for 58.
    assert q.invoice_header(CUSTOMER, 1) is None


def test_ambiguous_titles_are_surfaced_not_guessed():
    """'Believe' is three different songs by three different artists."""
    artists = {row["artist"] for row in q.exact_title_artists("Believe")}
    assert len(artists) > 1


def test_catalog_tools_expose_no_identity_parameter():
    for tool in (search_catalog, make_recommend_tracks(exclude_owned=True)):
        props = {p.lower() for p in tool.tool_call_schema.model_json_schema()["properties"]}
        assert not {"customer_id", "user_id", "account_id"} & props
