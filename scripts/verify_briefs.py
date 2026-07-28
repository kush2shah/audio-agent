"""Run the case investigator and fact-check every number it wrote.

    uv run python scripts/verify_briefs.py

The investigator writes prose for a human to act on, so the interesting question
isn't "did it finish" - it's "is any of this true". This pulls every figure out
of the briefs and checks it against the database.

Written after the first run claimed an album was "fully unowned" for a customer
who owned a track from it. The tool was right; the summary went further than the
tool. Nothing in the trace shows that, because writing a confident sentence looks
identical to writing a correct one.
"""

import re
import sys
import time

from dotenv import load_dotenv

load_dotenv()

from chinook_support import cases, queries  # noqa: E402
from chinook_support.db import query  # noqa: E402
from chinook_support.investigator import build_investigator  # noqa: E402

GREEN, RED, DIM, BOLD, RESET = "\033[32m", "\033[31m", "\033[2m", "\033[1m", "\033[0m"


def facts_for(customer_id: int) -> dict:
    totals = query(
        """SELECT COUNT(*) AS orders, ROUND(SUM(Total), 2) AS spend
           FROM Invoice WHERE CustomerId = ?""",
        (customer_id,),
    )[0]
    artists = query(
        """SELECT ar.Name AS artist, COUNT(*) AS n
           FROM Invoice i JOIN InvoiceLine il ON il.InvoiceId = i.InvoiceId
           JOIN Track t ON t.TrackId = il.TrackId
           JOIN Album al ON al.AlbumId = t.AlbumId
           JOIN Artist ar ON ar.ArtistId = al.ArtistId
           WHERE i.CustomerId = ? GROUP BY 1 ORDER BY n DESC""",
        (customer_id,),
    )
    return {"orders": totals["orders"], "spend": totals["spend"],
            "artists": {a["artist"]: a["n"] for a in artists}}


def album_ownership(customer_id: int, album: str) -> tuple[int, int] | None:
    rows = query(
        """SELECT COUNT(*) AS total,
                  SUM(CASE WHEN owned.TrackId IS NULL THEN 0 ELSE 1 END) AS owned
           FROM Track t JOIN Album al ON al.AlbumId = t.AlbumId
           LEFT JOIN (SELECT DISTINCT il.TrackId FROM Invoice i
                      JOIN InvoiceLine il ON il.InvoiceId = i.InvoiceId
                      WHERE i.CustomerId = ?) owned ON owned.TrackId = t.TrackId
           WHERE al.Title = ?""",
        (customer_id, album),
    )
    return (rows[0]["total"], rows[0]["owned"]) if rows and rows[0]["total"] else None


def check(brief: str, customer_id: int) -> list[tuple[bool, str]]:
    facts = facts_for(customer_id)
    results = []

    for m in re.finditer(r"(\d+)\s+orders\b", brief, re.I):
        claimed = int(m.group(1))
        results.append((claimed == facts["orders"],
                        f"orders: claimed {claimed}, actual {facts['orders']}"))

    # Only figures the brief labels as lifetime/total spend. Matching the first
    # currency value in the document instead flagged average-order figures as
    # wrong lifetime totals - a checker that cries wolf gets ignored just as fast
    # as one that rubber-stamps.
    for m in re.finditer(
        r"(?:lifetime|total)[^\n$]{0,30}\$\s?([\d,]+\.\d{2})"
        r"|\$\s?([\d,]+\.\d{2})[^\n.]{0,20}(?:lifetime|total spend)",
        brief, re.I,
    ):
        claimed = float((m.group(1) or m.group(2)).replace(",", ""))
        results.append((abs(claimed - facts["spend"]) < 0.01,
                        f"lifetime spend: claimed ${claimed}, actual ${facts['spend']}"))

    # "Iron Maiden (16 tracks)" style claims
    for artist, n in re.findall(r"\*\*([A-Z][\w' .&]+?)\*\*\s*\((\d+)\s*tracks?\)", brief):
        artist = artist.strip()
        actual = facts["artists"].get(artist)
        results.append((actual == int(n),
                        f"{artist}: claimed {n} tracks, actual {actual}"))

    # "own 2 of 18 tracks" - check it against the album named nearby.
    #
    # This used to append (True, ...) unconditionally, i.e. print a tick without
    # checking anything, inside a script whose entire purpose is checking. Worse
    # than not having the check, because it reported a pass.
    for match in re.finditer(
        r"\*([^*\n]{3,60}?)\*[^.\n]{0,120}?own[s]?\s+(\d+)\s+of\s+(\d+)\s+tracks",
        brief, re.I,
    ):
        album, owned, total = match.group(1).strip(" *—-–"), int(match.group(2)), int(match.group(3))
        counts = album_ownership(customer_id, album)
        if counts is None:
            results.append((False, f"'{album}': no such album in the catalog"))
        else:
            real_total, real_owned = counts
            results.append((
                (owned, total) == (real_owned, real_total),
                f"'{album}': claimed owns {owned}/{total}, actual {real_owned}/{real_total}",
            ))

    # The claim that went wrong the first time.
    for album in re.findall(r"\*([^*]+?)\*[^.]{0,80}?fully unowned", brief, re.I):
        counts = album_ownership(customer_id, album.strip())
        if counts:
            total, owned = counts
            results.append((owned == 0,
                            f"'{album.strip()}' claimed fully unowned: owns {owned}/{total}"))
    return results


def main() -> None:
    cases.reset()
    for customer_id in (58, 35):
        cases.open_case(customer_id, queries.support_rep_for(customer_id)["rep_id"],
                        "handoff: repeated_dead_ends")

    print(f"{DIM}running the investigator over the open case queue...{RESET}")
    started = time.time()
    agent = build_investigator()
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "Work the open case queue. Write a brief for each case."}]}
    )
    calls = [c["name"] for m in result["messages"] for c in (getattr(m, "tool_calls", None) or [])]
    print(f"{DIM}{time.time() - started:.0f}s, {len(result['messages'])} messages, "
          f"{len(calls)} tool calls{RESET}\n")

    open_cases = {c["case_id"]: c["customer_id"] for c in cases.open_cases_for(58) + cases.open_cases_for(35)}
    failures = 0

    for name, body in (result.get("files") or {}).items():
        text = body.get("content") if isinstance(body, dict) else str(body)
        case_id = int(re.search(r"case-(\d+)", name).group(1))
        customer_id = open_cases.get(case_id)
        if customer_id is None:
            continue
        print(f"{BOLD}{name}  (customer {customer_id}){RESET}")
        checks = check(text, customer_id)
        if not checks:
            # Not a pass. A brief with no verifiable numbers is a brief this
            # script cannot vouch for, and saying nothing is how a checker ends
            # up rubber-stamping.
            failures += 1
            print(f"  {RED}UNCHECKED{RESET} no verifiable claims found in this brief")
        for ok, detail in checks:
            failures += not ok
            print(f"  {GREEN + 'ok  ' if ok else RED + 'WRONG'}{RESET} {detail}")
        print()

    sys.exit(f"{failures} false claim(s)" if failures else "every checkable claim is true")


if __name__ == "__main__":
    main()
