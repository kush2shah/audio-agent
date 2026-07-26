"""Middleware - the things that wrap every turn, regardless of which tool ran.

Two here, doing very different jobs:

`handoff_to_human` is product behavior. A support bot that cannot recognize it is
failing is worse than no bot, because it traps the customer. This is the feature
a real store could not ship without.

`receipts` is observability. It lifts what the tools actually enforced onto the
run itself, so LangSmith can be queried by business rule rather than only by
latency and tokens.
"""

import json
import re
from typing import Any

from langchain.agents.middleware import (
    AgentState,
    ModelRequest,
    after_model,
    dynamic_prompt,
    hook_config,
)
from langchain.messages import AIMessage
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


def _root_run():
    """The trace's root run, walking up from wherever we're called.

    `get_current_run_tree()` inside a middleware hook returns that hook's own
    span (`receipts.after_model`). Metadata written there is real but useless -
    LangSmith's run filters match against root runs, so a receipt on a nested
    span can't be queried. Walk to the top.
    """
    run = get_current_run_tree()
    while run is not None and getattr(run, "parent_run", None) is not None:
        run = run.parent_run
    return run


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


@after_model
@hook_config(can_jump_to=["end"])
def handoff_to_human(state: AgentState, runtime: Runtime[Ctx]) -> dict[str, Any] | None:
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

    customer_id = runtime.context.customer_id
    rep = queries.support_rep_for(customer_id)
    reason = "customer_requested" if explicit else "repeated_dead_ends"

    if run := _root_run():
        run.add_metadata({"handoff": True, "handoff_reason": reason})

    if rep is None:  # no rep on file - still stop, don't keep looping
        return {
            "messages": [
                AIMessage(
                    "I'm not able to sort this out here, so I'm passing you to our "
                    "support team - someone will follow up with you directly."
                )
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

    return {"messages": [AIMessage(f"{opening}. {closing}")], "jump_to": "end"}


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
    run = _root_run()
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
