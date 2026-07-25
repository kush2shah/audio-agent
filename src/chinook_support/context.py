"""Who the agent is acting on behalf of.

This is the whole isolation mechanism, and it is deliberately tiny.

`customer_id` is supplied per-run, alongside the messages rather than inside them:

    agent.invoke({"messages": [...]}, context=Ctx(customer_id=35))

Messages are attacker-controlled - the customer types them. Context is
caller-controlled. In production `customer_id` comes from the authenticated
session, never from anything the client sent. Here it comes from the Studio
assistant config or the SDK call.

It is never a tool argument, so it is never something the model chooses.
See tools/account.py for why that matters.
"""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Ctx:
    """Runtime context, declared to create_agent() via `context_schema`.

    The env var is only a fallback so `langgraph dev` boots with something
    sane; every real run passes the value explicitly.
    """

    customer_id: int = int(os.getenv("DEMO_CUSTOMER_ID", "58"))
