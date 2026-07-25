"""Handoff and receipts logic, tested without a model.

The decision of when to give up on a customer is business logic, so it lives in
plain functions that can be tested directly rather than inside a hook that needs
an agent to exercise.
"""

import json

import pytest
from langchain.messages import AIMessage, HumanMessage, ToolMessage

from chinook_support import queries
from chinook_support.middleware import (
    CONSECUTIVE_DEAD_ENDS_BEFORE_HANDOFF,
    _asked_for_a_human,
    _consecutive_dead_ends,
    _tool_payloads,
)


def tool_msg(status: str, **constraints) -> ToolMessage:
    return ToolMessage(
        content=json.dumps(
            {"status": status, "data": [], "message": "", "next_actions": [],
             "constraints_applied": constraints}
        ),
        tool_call_id="x",
    )


def test_dead_ends_count_only_the_trailing_streak():
    payloads = _tool_payloads(
        {"messages": [tool_msg("not_available"), tool_msg("ok"), tool_msg("no_match")]}
    )
    assert _consecutive_dead_ends(payloads) == 1


def test_a_success_resets_the_streak():
    payloads = _tool_payloads(
        {"messages": [tool_msg("error"), tool_msg("not_available"), tool_msg("ok")]}
    )
    assert _consecutive_dead_ends(payloads) == 0


def test_three_dead_ends_reaches_the_handoff_threshold():
    payloads = _tool_payloads({"messages": [tool_msg("not_available")] * 3})
    assert _consecutive_dead_ends(payloads) >= CONSECUTIVE_DEAD_ENDS_BEFORE_HANDOFF


@pytest.mark.parametrize(
    "text,expected",
    [
        ("I want to speak to a human", True),
        ("can I talk to someone please", True),
        ("get me a representative", True),
        ("escalate this", True),
        ("what did I buy?", False),
        ("recommend me something", False),
        # A track called "Human" must not trigger a handoff.
        ("do you have any songs by Human League?", False),
    ],
)
def test_explicit_human_requests_are_recognized(text, expected):
    assert _asked_for_a_human({"messages": [HumanMessage(text)]}) is expected


def test_only_the_latest_human_message_counts():
    """Asking for a human once shouldn't hand off every later turn."""
    state = {
        "messages": [
            HumanMessage("I want to speak to a human"),
            AIMessage("handing you over"),
            HumanMessage("actually never mind, what did I buy?"),
        ]
    }
    assert _asked_for_a_human(state) is False


def test_malformed_tool_output_counts_as_a_dead_end():
    """A tool result that isn't JSON is a failure, not something to ignore."""
    payloads = _tool_payloads({"messages": [ToolMessage(content="boom", tool_call_id="x")]})
    assert payloads[0]["status"] == "error"


def test_every_customer_has_a_reachable_rep():
    """The handoff promises a named person, so every customer needs one."""
    for customer_id in (1, 35, 58):
        rep = queries.support_rep_for(customer_id)
        assert rep and rep["rep_name"] and rep["rep_email"]

    missing = [c for c in range(1, 60) if queries.support_rep_for(c) is None]
    assert not missing, f"customers with no support rep: {missing}"
