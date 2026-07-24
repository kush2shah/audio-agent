"""Workflow 1 - the customer's own account and purchase history.

Look at the signature the model sees:

    list_my_invoices(limit: int)

There is no `customer_id` parameter. `runtime` is injected by LangChain at call
time and is stripped from the schema sent to the model. So "show me customer 5's
invoices" is not a rule the model is asked to follow and might ignore - it is a
request the model has no way to express.
"""

from langchain.tools import ToolRuntime, tool

from .. import queries
from ..context import Ctx
from ..contracts import ToolResult, no_match, not_available, ok


@tool
def list_my_invoices(limit: int, runtime: ToolRuntime[Ctx]) -> ToolResult:
    """List the signed-in customer's past orders, newest first.

    Args:
        limit: How many orders to return. Use 5 unless the customer asks for more.
    """
    customer_id = runtime.context.customer_id
    rows = queries.invoices_for_customer(customer_id, max(1, min(limit, 25)))

    if not rows:
        return no_match(
            "This account has no orders yet.",
            next_actions=["Offer to help them find something to listen to."],
            customer_scope="authenticated_customer",
        )
    return ok(
        rows,
        f"{len(rows)} order(s) for the signed-in customer.",
        customer_scope="authenticated_customer",
    )


@tool
def get_my_invoice(invoice_id: int, runtime: ToolRuntime[Ctx]) -> ToolResult:
    """Show what was on one of the signed-in customer's orders.

    Args:
        invoice_id: The order number the customer is asking about.
    """
    customer_id = runtime.context.customer_id

    # Ownership is checked in the WHERE clause, not after the fact. There is no
    # moment where this function holds another customer's invoice in memory.
    header = queries.invoice_header(customer_id, invoice_id)
    if header is None:
        return not_available(
            f"Order {invoice_id} isn't on this account.",
            next_actions=["Offer to list the orders that are on this account."],
        )

    lines = queries.invoice_lines(invoice_id)
    return ok(
        lines,
        f"Order {invoice_id} from {header['date']}, {len(lines)} item(s), "
        f"total ${header['total']:.2f}.",
        customer_scope="authenticated_customer",
        ownership_verified=True,
    )
