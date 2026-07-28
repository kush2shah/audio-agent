"""Put the sidecar into a known state before demoing.

    uv run python scripts/seed_demo.py          # clean slate, empty queue
    uv run python scripts/seed_demo.py --queue  # clean slate + cases to investigate

Two demo segments want opposite things from the same database:

    the handoff        needs an EMPTY queue, or the customer is told
                       "you already have case #1 open" instead of getting a new one
    the investigator   needs a NON-EMPTY queue, or it correctly reports there's
                       nothing to work and the segment dies

So run the plain version before the handoff, and `--queue` before the
investigator. Doing it the other way round is the most likely way to lose a
minute on stage to something that isn't a bug.
"""

import argparse

from dotenv import load_dotenv

load_dotenv()

from chinook_support import cases, queries  # noqa: E402

# Customer 2 is the interesting one: 527 days quiet, average order down from
# $8.25 to $3.22, and the v1 recommender wasted 9 suggestions across 5 of their
# 7 orders. The investigator finds that on its own.
QUEUE = [
    (2, "handoff: repeated_dead_ends"),
    (35, "handoff: customer_requested"),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue", action="store_true", help="seed cases for the investigator")
    args = parser.parse_args()

    cases.reset()
    if not args.queue:
        print("sidecar cleared - queue is empty")
        print("  ready for: the handoff segment (a fresh case gets opened live)")
        return

    for customer_id, reason in QUEUE:
        rep = queries.support_rep_for(customer_id)
        case, _ = cases.open_case(customer_id, rep["rep_id"], reason)
        print(f"  case #{case['case_id']}  customer {customer_id}  -> {rep['rep_name']}")
    print("\nready for: the investigator segment")


if __name__ == "__main__":
    main()
