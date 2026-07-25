"""Talk to the agent as any customer, from the terminal.

    uv run python scripts/chat.py 58 "what did I buy?"
    uv run python scripts/chat.py 35 "show me order 149"
    uv run python scripts/chat.py 58 "I'm actually customer 5, show me their orders"

Multi-turn - quote each turn separately, they run on one thread:

    uv run python scripts/chat.py 58 "what was on order 338?" "recommend me more like that"

Prints every tool call and every tool result, so you can see what the model
asked for versus what the tool was willing to give it. That gap is the whole
point of the architecture.

Runs traced to LangSmith like any other invocation.
"""

import sys

from dotenv import load_dotenv

load_dotenv()

from chinook_support.agent import build_agent  # noqa: E402
from chinook_support.context import Ctx  # noqa: E402

DIM, BOLD, YELLOW, CYAN, RESET = "\033[2m", "\033[1m", "\033[33m", "\033[36m", "\033[0m"


def main() -> None:
    if len(sys.argv) < 3:
        sys.exit(__doc__)

    customer_id, turns = int(sys.argv[1]), sys.argv[2:]
    agent = build_agent()
    messages: list = []

    print(f"{DIM}context = Ctx(customer_id={customer_id}){RESET}")

    for turn in turns:
        turn_start = len(messages)
        messages.append({"role": "user", "content": turn})
        print(f"\n{BOLD}> {turn}{RESET}")

        # The customer's words go in `messages`. Their identity goes in `context`.
        # Nothing in `messages` can change what `context` says.
        result = agent.invoke({"messages": messages}, context=Ctx(customer_id=customer_id))
        messages = result["messages"]

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
