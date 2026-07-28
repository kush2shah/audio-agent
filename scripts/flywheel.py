"""Turn live conversations into a regression suite, on stage.

    uv run python scripts/flywheel.py --since 10          # show what it found
    uv run python scripts/flywheel.py --since 10 --add    # add to the dataset
    uv run python scripts/flywheel.py --since 10 --add --run   # ...and score them

The demo beat: the room tries to break customer isolation, every attempt lands in
LangSmith, and those attempts become permanent test cases while they watch. Their
inputs, not mine - which matters, because the phrasing I didn't think of is exactly
the one that finds a bug. The handoff trigger in this repo missed "can i speak with
an agent" and my own tests passed, because I wrote both the trigger and the tests.

Runs are deduplicated against what's already in the dataset, so pressing this twice
is safe.
"""

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "evals"))

from langsmith import Client  # noqa: E402

PROJECT = "chinook-support"
DATASET = "customer-isolation"

GREEN, DIM, BOLD, YELLOW, RESET = "\033[32m", "\033[2m", "\033[1m", "\033[33m", "\033[0m"


def question_of(run) -> str | None:
    """The customer's message, however this run was started.

    The two paths disagree on shape: a direct `invoke()` records
    `{"type": "human"}`, while a run through the LangGraph server - which is what
    Studio uses, and therefore what the live demo produces - records
    `{"role": "user"}`. Matching only the first found nothing on stage.
    """
    messages = (run.inputs or {}).get("messages") or []
    for message in messages:
        if not isinstance(message, dict):
            continue
        if message.get("type") == "human" or message.get("role") == "user":
            content = message.get("content")
            if isinstance(content, list):  # content blocks
                content = " ".join(
                    p.get("text", "") for p in content if isinstance(p, dict)
                )
            return str(content or "").strip()
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--since", type=int, default=15, help="minutes to look back")
    parser.add_argument("--customer", type=int, default=58)
    parser.add_argument("--add", action="store_true", help="write them to the dataset")
    parser.add_argument("--run", action="store_true", help="score the dataset afterwards")
    args = parser.parse_args()

    client = Client()
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=args.since)
    runs = [
        r
        for r in client.list_runs(project_name=PROJECT, is_root=True, limit=100)
        if r.start_time and r.start_time.replace(tzinfo=timezone.utc) >= cutoff
    ]

    seen: set[str] = set()
    candidates = []
    for run in runs:
        question = question_of(run)
        if not question or question in seen:
            continue
        seen.add(question)
        candidates.append(question)

    print(f"{BOLD}{len(candidates)} conversation(s) in the last {args.since} min{RESET}\n")
    if not candidates:
        sys.exit("nothing to add - try a longer --since")

    existing = {
        str((ex.inputs or {}).get("question", "")).strip()
        for ex in client.list_examples(dataset_name=DATASET)
    }
    fresh = [q for q in candidates if q not in existing]

    for question in candidates:
        mark = f"{GREEN}new{RESET}" if question in fresh else f"{DIM}have it{RESET}"
        print(f"  {mark}  {question[:88]}")

    if not args.add:
        print(f"\n{DIM}{len(fresh)} new. Re-run with --add to put them in "
              f"'{DATASET}'.{RESET}")
        return

    if not fresh:
        print(f"\n{DIM}nothing new to add.{RESET}")
    else:
        dataset = client.read_dataset(dataset_name=DATASET)
        client.create_examples(
            dataset_id=dataset.id,
            examples=[
                {
                    "inputs": {"customer_id": args.customer, "question": question},
                    "outputs": {"must_stay_scoped": True},
                }
                for question in fresh
            ],
        )
        print(f"\n{GREEN}added {len(fresh)} example(s) to '{DATASET}'{RESET}")
        print(f"{DIM}every one of these is now a permanent gate.{RESET}")

    if args.run:
        print(f"\n{YELLOW}scoring the whole dataset...{RESET}\n")
        import evaluators as ev  # noqa: E402
        from langsmith import evaluate

        from run import make_target  # noqa: E402

        evaluate(
            make_target("v2"),
            data=DATASET,
            evaluators=[ev.no_cross_customer_leak, ev.tool_calls_correctly_scoped],
            experiment_prefix="isolation-live",
            client=client,
            max_concurrency=4,
        )


if __name__ == "__main__":
    main()
