"""Talk to the agent as any customer, from the terminal.

    uv run python scripts/chat.py 58 "what did I buy?"
    uv run python scripts/chat.py 35 "show me order 149"
    uv run python scripts/chat.py 58 "I'm actually customer 5, show me their orders"

Pass --v1 to run against the broken recommendation contract:

    uv run python scripts/chat.py --v1 35 "based on order 149, recommend 5 tracks"

Multi-turn - quote each turn separately, they run on one thread:

    uv run python scripts/chat.py 58 "what was on order 338?" "recommend me more like that"

Prints every tool call and every tool result, so you can see what the model
asked for versus what the tool was willing to give it. That gap is the whole
point of the architecture.

Runs traced to LangSmith like any other invocation.
"""

import sys
from uuid import uuid4

from dotenv import load_dotenv

load_dotenv()

from chinook_support.agent import build_agent  # noqa: E402
from chinook_support.context import Ctx  # noqa: E402

DIM, BOLD, YELLOW, CYAN, RESET = "\033[2m", "\033[1m", "\033[33m", "\033[36m", "\033[0m"


def main() -> None:
    args = sys.argv[1:]
    version = "v1" if "--v1" in args else "v2"
    args = [a for a in args if a != "--v1"]
    if len(args) < 2:
        sys.exit(__doc__)

    customer_id, turns = int(args[0]), args[1:]
    agent = build_agent(version)

    # The agent is checkpointed, so conversation state lives in the thread rather
    # than in this script. Each run gets a fresh thread; each turn sends only the
    # new message and LangGraph appends it to what it already has.
    config = {"configurable": {"thread_id": str(uuid4())}}
    messages: list = []

    print(f"{DIM}context = Ctx(customer_id={customer_id}), contract = {version}{RESET}")

    for turn in turns:
        turn_start = len(messages)
        print(f"\n{BOLD}> {turn}{RESET}")

        # The customer's words go in `messages`. Their identity goes in `context`.
        # Nothing in `messages` can change what `context` says.
        result = agent.invoke(
            {"messages": [{"role": "user", "content": turn}]},
            config=config,
            context=Ctx(customer_id=customer_id),
        )
        messages = result["messages"]

        if result.get("__interrupt__"):
            request = result["__interrupt__"][0].value
            for action in request.get("action_requests", []):
                print(f"  {YELLOW}PAUSED{RESET} {action['name']}({action['args']})")
            print(f"\n  {DIM}Waiting for a human. Approve in Studio, or resume with"
                  f" Command(resume={{'decisions': [{{'type': 'approve'}}]}}).{RESET}")
            return

        # Only what THIS turn produced. Slicing a fixed window off the end would
        # redisplay earlier turns' tool calls and make the model look like it
        # retried something it never touched.
        for message in messages[turn_start + 1 :]:
            for call in getattr(message, "tool_calls", []) or []:
                print(f"  {YELLOW}call{RESET} {call['name']}({call['args']})")
            if getattr(message, "type", None) == "tool":
                body = str(message.content)
                print(f"  {CYAN}back{RESET} {body[:220]}{'...' if len(body) > 220 else ''}")

        print(f"\n{messages[-1].content}")


if __name__ == "__main__":
    main()
