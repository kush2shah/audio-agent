"""Workflow 3 - handing the customer to a person, and the only write in the app.

This tool is the *model's* escalation path. The middleware in middleware.py is
the deterministic one. They are deliberately different:

    middleware  fires on an explicit request or repeated dead ends.
                A guarantee - it does not depend on the model deciding well.

    this tool   fires when the model senses trouble a pattern can't see:
                frustration, confusion, someone rephrasing the same thing.
                A nicety - useful, but never something to rely on.

The floor doesn't depend on the model; the ceiling does. That split matters here
more than most places, because "give up and fetch a human" is the one action a
helpfulness-trained model is least inclined to take - and the moment it's least
reliable is exactly when the conversation has already gone wrong.

Because it writes, it's gated by HumanInTheLoopMiddleware: the model proposes,
a person approves, and only then does anything hit the database.
"""

from langchain.tools import ToolRuntime, tool

from .. import cases, queries
from ..context import Ctx
from ..contracts import ToolResult, error, ok


@tool
def escalate_to_human(reason: str, runtime: ToolRuntime[Ctx]) -> ToolResult:
    """Open a support case so a human colleague can take over.

    Use when the customer seems frustrated, when you've tried something twice
    without success, or when they need something you have no tool for.

    Args:
        reason: One sentence a human colleague can pick this up from.
    """
    customer_id = runtime.context.customer_id
    rep = queries.support_rep_for(customer_id)
    if rep is None:
        return error("I couldn't reach the support team just now.")

    case, created = cases.open_case(customer_id, rep["rep_id"], reason)

    if not created:
        return ok(
            [case],
            f"This customer already has case #{case['case_id']} open with "
            f"{rep['rep_name']}. Reassure them rather than opening another.",
            case_id=case["case_id"],
            already_open=True,
        )

    return ok(
        [case],
        f"Case #{case['case_id']} opened with {rep['rep_name']} "
        f"({rep['rep_email']}). Tell the customer who will follow up.",
        case_id=case["case_id"],
        rep_name=rep["rep_name"],
        already_open=False,
    )
