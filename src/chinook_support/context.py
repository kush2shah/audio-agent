"""Who the agent is acting on behalf of.

This is the whole isolation mechanism, and it is deliberately tiny.

`customer_id` arrives from the runtime - in production, from your authenticated
session; here, from Studio's config panel. It is never a tool argument, so it is
never something the model chooses. See tools/account.py for why that matters.
"""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Ctx:
    """Runtime context. Studio surfaces this as a config field."""

    customer_id: int = int(os.getenv("DEMO_CUSTOMER_ID", "58"))
