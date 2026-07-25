"""Show, side by side, what we recommended and what they already own.

    uv run python scripts/proof.py 35 149
    uv run python scripts/proof.py 58 338

Runs the v1 and v2 recommendation queries directly - no model, no tokens, no
agent. Every overlap is printed with the order it was originally bought on, so
"they already own this" is a fact on screen rather than a claim.
"""

import sys

from chinook_support import queries
from chinook_support.db import query

RED, GREEN, DIM, BOLD, RESET = "\033[31m", "\033[32m", "\033[2m", "\033[1m", "\033[0m"


def bought_on(customer_id: int, track_id: int) -> str:
    rows = query(
        """SELECT i.InvoiceId AS invoice_id, DATE(i.InvoiceDate) AS date
           FROM Invoice i
           JOIN InvoiceLine il ON il.InvoiceId = i.InvoiceId
           WHERE i.CustomerId = ? AND il.TrackId = ?
           ORDER BY i.InvoiceDate""",
        (customer_id, track_id),
    )
    return ", ".join(f"order {r['invoice_id']} ({r['date']})" for r in rows)


def show(customer_id: int, seed_invoice: int, version: str, exclude_owned: bool) -> int:
    seed = queries.invoice_seed_profile(seed_invoice)
    owned = queries.owned_track_ids(customer_id)
    rows = queries.recommend(customer_id, seed, None, None, 5, exclude_owned)

    print(f"\n{BOLD}{version}{RESET}")
    leaks = 0
    for row in rows:
        if row["track_id"] in owned:
            leaks += 1
            print(
                f"  {RED}ALREADY OWNED{RESET}  {row['track'][:38]:40}"
                f" {DIM}bought on {bought_on(customer_id, row['track_id'])}{RESET}"
            )
        else:
            print(f"  {GREEN}new to them  {RESET}  {row['track'][:38]:40}")
    return leaks


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    customer_id, seed_invoice = int(sys.argv[1]), int(sys.argv[2])

    lines = queries.invoice_lines(seed_invoice)
    owned = queries.owned_track_ids(customer_id)
    print(f"{BOLD}Customer {customer_id}{RESET} owns {len(owned)} tracks across their order history.")
    print(f"Seeding recommendations from order {seed_invoice}: "
          f"{', '.join(line['track'] for line in lines)}")

    v1 = show(customer_id, seed_invoice, "v1 - ranked by store-wide sales", False)
    v2 = show(customer_id, seed_invoice, "v2 - same query, plus the ownership anti-join", True)

    print(f"\n{BOLD}v1 wasted {v1} of 5 slots. v2 wasted {v2}.{RESET}\n")


if __name__ == "__main__":
    main()
