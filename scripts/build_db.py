"""Rebuild data/chinook.db from data/data.sql, then assert it looks right.

Run: uv run python scripts/build_db.py

The integrity gate exists so a silent data change can never quietly break the
demo. Every number below is a fixture the demo script depends on.
"""

import sqlite3
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SQL = ROOT / "data" / "data.sql"
DB = ROOT / "data" / "chinook.db"

# (label, query, expected)
CHECKS = [
    ("customers", "SELECT COUNT(*) FROM Customer", 59),
    ("invoices", "SELECT COUNT(*) FROM Invoice", 412),
    ("invoice lines", "SELECT COUNT(*) FROM InvoiceLine", 2240),
    ("tracks", "SELECT COUNT(*) FROM Track", 3503),
    ("genres", "SELECT COUNT(*) FROM Genre", 25),
    # Every customer has an assigned rep - the escalation workflow relies on it.
    ("customers with a rep", "SELECT COUNT(*) FROM Customer WHERE SupportRepId IS NOT NULL", 59),
    # The headline demo fixture: customer 58, invoice 338, four Gene Krupa Jazz tracks.
    ("invoice 338 owner", "SELECT CustomerId FROM Invoice WHERE InvoiceId = 338", 58),
    ("invoice 338 line count", "SELECT COUNT(*) FROM InvoiceLine WHERE InvoiceId = 338", 4),
    # The staged failure depends on customer 58 owning exactly these Gene Krupa tracks.
    (
        "invoice 338 track ids",
        "SELECT GROUP_CONCAT(TrackId) FROM (SELECT TrackId FROM InvoiceLine "
        "WHERE InvoiceId = 338 ORDER BY TrackId)",
        "633,635,637,639",
    ),
    # ...and on two of them ranking in the artist's top four by global sales.
    (
        "gene krupa top 4 by sales",
        """SELECT GROUP_CONCAT(TrackId) FROM (
             SELECT t.TrackId, COUNT(il.InvoiceLineId) buys
             FROM Track t
             JOIN Album al ON al.AlbumId = t.AlbumId
             JOIN Artist ar ON ar.ArtistId = al.ArtistId
             LEFT JOIN InvoiceLine il ON il.TrackId = t.TrackId
             WHERE ar.Name = 'Gene Krupa'
             GROUP BY t.TrackId ORDER BY buys DESC, t.TrackId LIMIT 4)""",
        "625,626,635,639",
    ),
]


def build() -> None:
    if not SQL.exists():
        sys.exit(f"missing {SQL}")
    DB.unlink(missing_ok=True)
    proc = subprocess.run(
        ["sqlite3", str(DB)],
        stdin=SQL.open("rb"),
        capture_output=True,
    )
    if proc.returncode != 0:
        sys.exit(f"sqlite3 load failed:\n{proc.stderr.decode()[:2000]}")


def verify() -> int:
    con = sqlite3.connect(DB)
    failures = 0
    for label, query, expected in CHECKS:
        actual = con.execute(query).fetchone()[0]
        ok = actual == expected
        failures += not ok
        print(f"{'ok  ' if ok else 'FAIL'}  {label}: {actual}" + ("" if ok else f" (expected {expected})"))

    # Referential integrity, and no invoice whose lines don't sum to its total.
    violations = con.execute("PRAGMA foreign_key_check").fetchall()
    print(f"{'ok  ' if not violations else 'FAIL'}  foreign key violations: {len(violations)}")
    failures += bool(violations)

    mismatched = con.execute(
        """SELECT COUNT(*) FROM Invoice i WHERE ROUND(i.Total, 2) <> (
             SELECT ROUND(COALESCE(SUM(il.UnitPrice * il.Quantity), 0), 2)
             FROM InvoiceLine il WHERE il.InvoiceId = i.InvoiceId)"""
    ).fetchone()[0]
    print(f"{'ok  ' if not mismatched else 'FAIL'}  invoices with total mismatch: {mismatched}")
    failures += bool(mismatched)

    con.close()
    return failures


if __name__ == "__main__":
    build()
    print(f"built {DB.relative_to(ROOT)}\n")
    n = verify()
    if n:
        sys.exit(f"\n{n} check(s) failed")
    print("\nall checks passed")
