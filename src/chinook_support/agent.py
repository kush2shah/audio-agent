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

from .context import Ctx
from .tools.account import get_my_invoice, list_my_invoices

MODEL = os.getenv("CHINOOK_MODEL", "anthropic:claude-sonnet-4-6")

SYSTEM_PROMPT = """You are the customer support assistant for Chinook Records, an \
online music store.

You can look up the signed-in customer's order history and the contents of their \
individual orders.

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
- Be warm and concise. You're a record store, not a bank."""


def build_agent():
    return create_agent(
        model=MODEL,
        tools=[list_my_invoices, get_my_invoice],
        system_prompt=SYSTEM_PROMPT,
        context_schema=Ctx,
    )


graph = build_agent()
