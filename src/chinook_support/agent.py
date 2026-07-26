"""The agent itself.

`create_agent()` builds a LangGraph state machine: model -> tools -> model, with
middleware hooks around each step. We are not hand-writing a router graph, because
this is a short-horizon support conversation with a small typed tool surface -
exactly what the prebuilt loop is for.

Note what the system prompt does *not* do. It does not say "only show the customer
their own data" or "don't recommend things they already own". Those are guarantees,
and guarantees belong in the tool layer where they can be tested. The prompt only
governs tone and how to talk about tool results.
"""

import os

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver

from .context import Ctx
from .middleware import handoff_to_human, receipts, with_assigned_rep
from .tools.account import get_my_invoice, list_my_invoices
from .tools.catalog import make_recommend_tracks, search_catalog
from .tools.support import escalate_to_human

MODEL = os.getenv("CHINOOK_MODEL", "anthropic:claude-sonnet-4-6")

# Sentinel: "you didn't say, so bring your own checkpointer". Distinct from an
# explicit None, which means "the platform is providing persistence".
_OWN_CHECKPOINTER = object()

SYSTEM_PROMPT = """You are the customer support assistant for Chinook Records, an \
online music store.

You can look up the signed-in customer's order history, the contents of their \
individual orders, search the catalog, and recommend music based on what they've \
bought before.

Rules:
- Every factual claim about orders or the catalog must come from a tool result. \
Never answer from memory about what this store sells or what this customer bought.
- Tool results carry a `status`. Treat it as authoritative:
  - `ok` - use the data.
  - `no_match` - say plainly that there's nothing, then offer a next step.
  - `ambiguous` - ask the customer which one they meant. Do not pick for them.
  - `not_available` - tell them you can't pull that up on their account. Do not \
speculate about why, and do not confirm or deny that the record exists.
  - `error` - apologize briefly and suggest they try again.
- Be warm and concise. You're a record store, not a bank.
- You CAN hand a customer to their assigned human rep, and this happens \
automatically when they ask. Never tell a customer you're unable to transfer them \
or that no human is available - that isn't true.
- Call `escalate_to_human` when someone seems frustrated, when you've tried the \
same thing twice without success, or when they need something you have no tool \
for. Opening a case is a normal, good outcome - not a failure.
- Always say something to the customer in the same turn you call \
`escalate_to_human` - tell them you're getting someone. Approval can take a \
moment, and silence from you is indistinguishable from the system being broken.
- If an escalation doesn't go through, don't imply you changed your mind or that \
they don't need help. Give them their rep's email so they aren't left with \
nothing."""


def build_agent(contract_version: str = "v2", *, checkpointer: object = _OWN_CHECKPOINTER):
    """Build the agent against one version of the recommendation contract.

    v1 and v2 differ only in the SQL behind `recommend_tracks`. Same model, same
    prompt, same tool names and signatures. Exporting both as separate graphs
    beats flipping an environment variable: Studio can show them side by side,
    no restart is needed mid-demo, and an experiment can't accidentally pick up
    the wrong one.
    """
    agent = create_agent(
        model=MODEL,
        tools=[
            list_my_invoices,
            get_my_invoice,
            search_catalog,
            make_recommend_tracks(exclude_owned=contract_version == "v2"),
            escalate_to_human,
        ],
        system_prompt=SYSTEM_PROMPT,
        context_schema=Ctx,
        middleware=[
            # Reads are safe and run unattended. The one tool that writes pauses
            # for a person. Not "the model is untrustworthy" - it's that writes
            # are where mistakes become other people's problems.
            HumanInTheLoopMiddleware(
                interrupt_on={
                    "escalate_to_human": {"allowed_decisions": ["approve", "reject"]},
                },
                description_prefix="Opening a support case - needs approval",
            ),
            # Puts the customer's actual rep in the prompt, so the model never
            # has to invent a name or an email.
            with_assigned_rep,
            # handoff_to_human can end the turn; receipts only observes. Both run
            # after the model, so both see the completed tool results for the turn.
            handoff_to_human,
            receipts,
        ],
        # Interrupts need somewhere to keep the conversation while it waits, but
        # who provides that depends on how the agent is being run:
        #
        #   direct invoke()      we must supply one, or interrupts can't pause
        #   langgraph dev/API    the platform supplies it, and passing our own
        #                        is a hard startup error
        #
        # So the module-level graphs below pass None, and everything else gets
        # an InMemorySaver. Fine for a demo; production wants Postgres.
        checkpointer=InMemorySaver() if checkpointer is _OWN_CHECKPOINTER else checkpointer,
    )
    # Stamped on the root run at invoke time, so `contract_version` is filterable
    # across whole traces - which is how you find every v1 run in production
    # after you learn v1 was wrong.
    return agent.with_config(metadata={"contract_version": contract_version})


# Exported for langgraph.json. checkpointer=None because the LangGraph server
# provides persistence itself and rejects a graph that brings its own.
graph_v1 = build_agent("v1", checkpointer=None)  # ships the bug, kept for the demo
graph_v2 = build_agent("v2", checkpointer=None)  # the fix
graph = graph_v2
