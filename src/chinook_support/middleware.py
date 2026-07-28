"""Middleware - the things that wrap every turn, regardless of which tool ran.

Two here, doing very different jobs:

`handoff_to_human` is product behavior. A support bot that cannot recognize it is
failing is worse than no bot, because it traps the customer. This is the feature
a real store could not ship without.

`receipts` is observability. It lifts what the tools actually enforced onto the
run itself, so LangSmith can be queried by business rule rather than only by
latency and tokens.
"""

import asyncio
import json
import re
from typing import Any

from langchain.agents.middleware import (
    AgentMiddleware,
    AgentState,
    ModelRequest,
    after_model,
    dynamic_prompt,
    hook_config,
)
from langchain.messages import AIMessage, ToolMessage
from langgraph.runtime import Runtime
from langsmith import get_current_run_tree

from . import cases, queries
from .context import Ctx

# A tool saying "I have nothing for you" is not an error - it is a normal answer.
# But several in a row means the customer is asking for something this agent
# structurally cannot do, and continuing to try wastes their time.
DEAD_END_STATUSES = {"not_available", "no_match", "error"}
CONSECUTIVE_DEAD_ENDS_BEFORE_HANDOFF = 3

# Matching a customer's request for a person. Deliberately pattern-based rather
# than sentiment-based: "are they frustrated" is a judgement call, "did they ask
# for a human" is a fact.
#
# The first version of this was a list of literal phrases and it missed
# "can I speak with an agent" - it had "speak to" but not "speak with", and no
# "agent" at all. Worth remembering that the natural phrasing is the one you
# didn't write down.
HUMAN_REQUESTED = re.compile(
    r"""
    (speak|talk|chat|connect|transfer|put|get)               # the ask
    \s+(me|us)?\s*(to|with|through)?\s*(a|an|the)?\s*        # optional glue
    (human|person|agent|someone|somebody|rep\b|representative|advisor|staff)
    | real\s+(person|human)
    | customer\s+service
    | (your|a)\s+(manager|supervisor)
    | escalate
    | human\s+support
    """,
    re.IGNORECASE | re.VERBOSE,
)


def _receipt_target():
    """Where a receipt can actually be written from inside a hook.

    Not the trace root, despite two attempts to make it so:

      - `get_current_run_tree()` returns the *hook's own span*, e.g.
        `HandoffToHuman.after_model`.
      - Walking up via `parent_run` doesn't work: on a reconstructed tree that
        attribute is None. Only `parent_run_id` and `trace_id` are populated,
        and neither gives you the object.
      - Patching the root afterwards with `client.update_run(trace_id, ...)`
        returns 409 Conflict once the run has completed.

    So receipts live on the middleware span. They are still filterable -
    `eq(metadata_key, "handoff_reason")` finds them - but the matches are spans,
    not root runs, and anything reading `run.extra["metadata"]` off a root run
    will find nothing. See scripts/audit_handoffs.py, which had exactly that bug.

    Anything knowable before the run starts belongs on the root instead, set via
    `agent.with_config(metadata=...)` - that's how `contract_version` gets there.
    """
    return get_current_run_tree()


def _tool_payloads(state: AgentState) -> list[dict]:
    """Every tool result in the conversation, oldest first."""
    payloads = []
    for message in state["messages"]:
        if getattr(message, "type", None) != "tool":
            continue
        try:
            payloads.append(json.loads(message.content))
        except (json.JSONDecodeError, TypeError):
            payloads.append({"status": "error"})
    return payloads


def _consecutive_dead_ends(payloads: list[dict]) -> int:
    count = 0
    for payload in reversed(payloads):
        if payload.get("status") in DEAD_END_STATUSES:
            count += 1
        else:
            break
    return count


HANDOFF_MARKER = "⁠"  # zero-width, invisible to the customer


def _already_handed_off(state: AgentState) -> bool:
    """Did we already hand this conversation over?

    Marked with a zero-width character rather than by matching the text, so
    rewording the handoff message can't silently break the check.
    """
    for message in state["messages"]:
        if getattr(message, "type", None) == "ai" and HANDOFF_MARKER in str(message.content):
            return True
    return False


def _settle_pending_tool_calls(state: AgentState) -> list:
    """Close out any tool call the model proposed but that will now never run.

    Ending the turn from `after_model` skips the tools node, which leaves an
    AIMessage holding a tool_call with no matching ToolMessage. Anthropic rejects
    that history on the *next* request:

        tool_use ids were found without tool_result blocks

    So the turn appears to succeed and the conversation is bricked from then on.
    Every pending call needs an answer before we jump.
    """
    last = state["messages"][-1] if state["messages"] else None
    pending = getattr(last, "tool_calls", None) or []
    return [
        ToolMessage(
            content="Not run - this conversation was handed to a human colleague.",
            tool_call_id=call["id"],
            name=call["name"],
        )
        for call in pending
    ]


def _asked_for_a_human(state: AgentState) -> bool:
    for message in reversed(state["messages"]):
        if getattr(message, "type", None) == "human":
            return bool(HUMAN_REQUESTED.search(str(message.content)))
    return False


def prompt_with_rep(base: str, customer_id: int) -> str:
    """Append this customer's real rep details to the system prompt.

    Added after the model invented a support email address. The prompt told it to
    give out the rep's contact details, but nothing ever put those details in
    context - on the rejected-escalation path the escalation tool never runs - so
    it produced a plausible-looking address that doesn't exist.

    The instruction wasn't wrong, it was unsupported. A model asked for a fact it
    doesn't have will supply something shaped like that fact. Rewording the
    instruction would have produced a differently-worded invention.
    """
    rep = queries.support_rep_for(customer_id)
    if rep is None:
        return base
    return (
        f"{base}\n\nThis customer's assigned rep is {rep['rep_name']} "
        f"({rep['rep_title']}), {rep['rep_email']}. Use these exact details when "
        f"referring to their rep - never guess a name or an address."
    )


@dynamic_prompt
def with_assigned_rep(request: ModelRequest) -> str:
    return prompt_with_rep(request.system_prompt or "", request.runtime.context.customer_id)


def _handoff(state: AgentState, runtime: Runtime[Ctx]) -> dict[str, Any] | None:
    """Stop the loop and route to the customer's own support rep.

    Fires when the customer asks for a person, or when the agent has hit several
    dead ends in a row and is clearly not converging.

    This is the middleware a real store could not ship without. An agent that
    keeps cheerfully trying is not being helpful - it is holding the customer
    hostage to its own limitations.
    """
    payloads = _tool_payloads(state)
    dead_ends = _consecutive_dead_ends(payloads)
    explicit = _asked_for_a_human(state)

    if not explicit and dead_ends < CONSECUTIVE_DEAD_ENDS_BEFORE_HANDOFF:
        return None

    # Already handed off on this conversation - don't do it again every turn.
    # The dead-end streak is computed over the whole history, so without this the
    # customer gets handed off on every subsequent message.
    if _already_handed_off(state):
        return None

    customer_id = runtime.context.customer_id
    rep = queries.support_rep_for(customer_id)
    reason = "customer_requested" if explicit else "repeated_dead_ends"

    if run := _receipt_target():
        run.add_metadata({"handoff": True, "handoff_reason": reason})

    if rep is None:  # no rep on file - still stop, don't keep looping
        return {
            "messages": [
                *_settle_pending_tool_calls(state),
                AIMessage(
                    "I'm not able to sort this out here, so I'm passing you to our "
                    "support team - someone will follow up with you directly."
                    + HANDOFF_MARKER
                ),
            ],
            "jump_to": "end",
        }

    # Open the case here rather than promising one. An earlier version told the
    # customer "they'll have this conversation in front of them" and wrote
    # nothing - a warm, specific, completely false promise.
    #
    # This write is not gated by human approval, unlike escalate_to_human, and
    # the distinction is the point: approval is required when the *model*
    # decides to escalate, not when the *customer* asks. There is no judgement
    # here for a reviewer to second-guess.
    case, created = cases.open_case(customer_id, rep["rep_id"], f"handoff: {reason}")

    if not created:
        opening = f"You already have case #{case['case_id']} open with {rep['rep_name']}"
        closing = "They have your history and will pick this up."
    else:
        opening = (
            "Of course" if explicit
            else "I'm not finding what you need, and I don't want to keep you going in circles"
        )
        opening += (
            f" - I've opened case #{case['case_id']} with {rep['rep_name']}, "
            f"{rep['rep_title'].lower()} for your account"
        )
        closing = (
            f"They'll see this whole conversation and usually reply within one "
            f"business day. If it's urgent, reach them directly at {rep['rep_email']}."
        )

    return {
        "messages": [
            *_settle_pending_tool_calls(state),
            AIMessage(f"{opening}. {closing}{HANDOFF_MARKER}"),
        ],
        "jump_to": "end",
    }


class HandoffToHuman(AgentMiddleware):
    """Wraps `_handoff` with both a sync and an async hook.

    Middleware hooks run directly in the event loop - unlike tools, which
    LangGraph runs in a threadpool - so a synchronous SQLite write here blocks
    the whole server. The LangGraph dev server rejects it outright:

        BlockingError: Blocking call to sqlite3.Connection.execute

    Defining only the async hook is not an option either: sync `.invoke()` then
    fails with "Synchronous implementation of ... is not available", which would
    break every test and script. So both exist, and the async one moves the write
    off the loop.
    """

    @hook_config(can_jump_to=["end"])
    def after_model(self, state: AgentState, runtime: Runtime[Ctx]) -> dict[str, Any] | None:
        return _handoff(state, runtime)

    @hook_config(can_jump_to=["end"])
    async def aafter_model(self, state: AgentState, runtime: Runtime[Ctx]) -> dict[str, Any] | None:
        return await asyncio.to_thread(_handoff, state, runtime)


handoff_to_human = HandoffToHuman()


@after_model
def receipts(state: AgentState, runtime: Runtime[Ctx]) -> None:
    """Stamp what the tools actually enforced onto the LangSmith run.

    Every tool returns `constraints_applied` describing the guarantees it applied.
    This lifts those onto the run as flat, top-level metadata keys.

    Flat and scalar on purpose: LangSmith's filter grammar matches key/value pairs
    (`eq(metadata_key, "excluded_previously_purchased")`) and has no nested-path
    syntax, so a nested dict would not be queryable at all.

    What that buys - questions you can now ask of production traffic:

        every run where the owned-track filter did not apply
        every run where a tool refused because a record wasn't on the account

    Same field the evaluators assert against, so "did the filter run?" is one
    fact with three consumers: model-readable, human-filterable, machine-assertable.
    """
    run = _receipt_target()
    if run is None:
        return None

    payloads = _tool_payloads(state)
    receipt: dict[str, Any] = {}
    statuses = []
    for payload in payloads:
        statuses.append(payload.get("status", "unknown"))
        for key, value in (payload.get("constraints_applied") or {}).items():
            # Everything is stringified. LangSmith's metadata filter compares
            # values as strings, so a Python bool stored as `True` will never
            # match eq(metadata_value, "true") - it silently returns nothing,
            # which is the worst kind of broken for a filter.
            receipt[key] = "null" if value is None else str(value).lower()

    if statuses:
        receipt["tool_statuses"] = ",".join(statuses)
        receipt["dead_end_streak"] = str(_consecutive_dead_ends(payloads))
    run.add_metadata(receipt)
    return None
