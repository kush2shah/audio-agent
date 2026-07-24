"""The isolation boundary, tested at the layer where it's actually enforced.

Two kinds of test here:

1. Schema tests - prove the model has no way to *express* a request for someone
   else's data. This is the structural argument.
2. Behavior tests - prove that even if it could, the query wouldn't return it.

Neither involves a model. Isolation that depends on a model behaving well isn't
isolation, so it shouldn't need a model to test.
"""

import pytest

from chinook_support import queries
from chinook_support.db import query
from chinook_support.tools.account import get_my_invoice, list_my_invoices

CUSTOMER = 58
THEIR_INVOICE = 338
SOMEONE_ELSES_INVOICE = 1  # customer 2's

# Anything that would let a model name a customer other than the signed-in one.
FORBIDDEN = {"customer_id", "customerid", "customer", "user_id", "userid", "account_id"}


@pytest.mark.parametrize("tool", [list_my_invoices, get_my_invoice], ids=lambda t: t.name)
def test_model_facing_schema_has_no_identity_parameter(tool):
    """The load-bearing test for the whole demo.

    `tool_call_schema` is what actually gets sent to the model - `args_schema`
    still contains the injected runtime. If this test ever fails, someone has
    added a parameter the model could fill with another customer's ID.
    """
    params = tool.tool_call_schema.model_json_schema()["properties"]
    assert not FORBIDDEN & {p.lower() for p in params}, (
        f"{tool.name} exposes an identity parameter to the model: {list(params)}"
    )


def test_runtime_is_not_exposed_to_the_model():
    """The injected runtime must not leak into the model-facing schema either."""
    for tool in (list_my_invoices, get_my_invoice):
        assert "runtime" not in tool.tool_call_schema.model_json_schema()["properties"]


def test_own_invoice_is_readable():
    assert queries.invoice_header(CUSTOMER, THEIR_INVOICE) is not None


def test_foreign_invoice_is_not_readable():
    assert queries.invoice_header(CUSTOMER, SOMEONE_ELSES_INVOICE) is None


def test_foreign_and_nonexistent_are_indistinguishable():
    """A customer must not be able to probe for the existence of other people's
    records by noticing that "not yours" and "no such order" behave differently."""
    foreign = queries.invoice_header(CUSTOMER, SOMEONE_ELSES_INVOICE)
    nonexistent = queries.invoice_header(CUSTOMER, 999_999)
    assert foreign == nonexistent is None


def test_no_customer_can_read_another_customers_invoice():
    """Sweep the whole dataset rather than trusting one hand-picked example."""
    owners = {
        row["invoice_id"]: row["customer_id"]
        for row in query("SELECT InvoiceId AS invoice_id, CustomerId AS customer_id FROM Invoice")
    }
    # Spot-check every invoice against a customer who doesn't own it.
    leaks = [
        invoice_id
        for invoice_id, owner in owners.items()
        if queries.invoice_header(owner + 1 if owner < 59 else 1, invoice_id) is not None
    ]
    assert not leaks, f"{len(leaks)} invoice(s) readable by a non-owner: {leaks[:5]}"
