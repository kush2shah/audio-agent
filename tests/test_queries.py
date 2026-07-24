"""Data-layer tests. No model, no agent, no API key, no tokens spent.

These are the tests that actually guard the isolation boundary. Everything above
this layer is a convenience wrapper around it.
"""

from chinook_support import queries

CUSTOMER = 58
THEIR_INVOICE = 338
SOMEONE_ELSES_INVOICE = 1  # belongs to customer 2


def test_invoices_are_scoped_to_the_customer():
    rows = queries.invoices_for_customer(CUSTOMER, 25)
    assert rows, "customer 58 should have orders"
    assert THEIR_INVOICE in {r["invoice_id"] for r in rows}


def test_own_invoice_is_visible():
    assert queries.invoice_header(CUSTOMER, THEIR_INVOICE) is not None


def test_someone_elses_invoice_is_not_visible():
    assert queries.invoice_header(CUSTOMER, SOMEONE_ELSES_INVOICE) is None


def test_nonexistent_invoice_is_indistinguishable_from_someone_elses():
    """Both return None, so the customer cannot probe for other people's records."""
    assert queries.invoice_header(CUSTOMER, 999_999) is None
    assert queries.invoice_header(CUSTOMER, SOMEONE_ELSES_INVOICE) is None


def test_invoice_338_is_the_demo_fixture():
    lines = queries.invoice_lines(THEIR_INVOICE)
    assert [line["track_id"] for line in lines] == [633, 635, 637, 639]
    assert {line["artist"] for line in lines} == {"Gene Krupa"}
    assert {line["genre"] for line in lines} == {"Jazz"}


def test_owned_tracks_include_the_seed_invoice():
    owned = queries.owned_track_ids(CUSTOMER)
    assert {633, 635, 637, 639} <= owned
    assert len(owned) == 38
