"""The tool result envelope.

Every tool returns this same shape whether it succeeded or failed.

The design argument, in one line: *structured exit codes let an agent recover;
`None` makes it guess.* A tool that returns an empty list on "not found" and an
empty list on "you're not allowed to see that" has told the model nothing, and the
model will paper over the gap with fluent prose.

`constraints_applied` is the field most tool designs omit. It records which
invariants the *tool* enforced, which makes those guarantees visible in a
LangSmith trace and assertable in an eval - instead of something you hope the
model respected.
"""

from typing import Any, Literal, TypedDict

Status = Literal["ok", "no_match", "ambiguous", "not_available", "error"]


class ToolResult(TypedDict):
    status: Status
    data: list[dict[str, Any]]
    message: str
    next_actions: list[str]
    constraints_applied: dict[str, Any]


def _result(
    status: Status,
    message: str,
    data: list[dict] | None = None,
    next_actions: list[str] | None = None,
    constraints: dict | None = None,
) -> ToolResult:
    return ToolResult(
        status=status,
        data=data or [],
        message=message,
        next_actions=next_actions or [],
        constraints_applied=constraints or {},
    )


def ok(data: list[dict], message: str, **constraints: Any) -> ToolResult:
    return _result("ok", message, data=data, constraints=constraints)


def no_match(message: str, next_actions: list[str] | None = None, **constraints: Any) -> ToolResult:
    """The query was valid and the answer is genuinely empty."""
    return _result("no_match", message, next_actions=next_actions, constraints=constraints)


def ambiguous(candidates: list[dict], message: str, **constraints: Any) -> ToolResult:
    """Several things match. Ask rather than silently picking one."""
    return _result(
        "ambiguous",
        message,
        data=candidates,
        next_actions=["Ask the customer which of these they meant."],
        constraints=constraints,
    )


def not_available(message: str, next_actions: list[str] | None = None) -> ToolResult:
    """The record does not exist, or exists but is not this customer's.

    Deliberately one status for both cases. If "not yours" and "no such invoice"
    were distinguishable, a customer could probe for the existence of other
    people's records by watching which error came back.
    """
    return _result("not_available", message, next_actions=next_actions)


def error(message: str) -> ToolResult:
    """Something broke. The message is sanitized before it reaches the model."""
    return _result("error", message)
