"""Support cases - the only thing this application writes.

Deliberately a separate database from Chinook. The catalog is the store's system
of record and the agent opens it read-only; cases are ours. If this demo went
wrong in the worst possible way, the blast radius is a file we created.

Idempotency is enforced by a UNIQUE constraint, not by the model remembering
whether it already opened a case. A model-generated request id would be exactly
as reliable as the model, which is the thing we're trying not to depend on.
"""

import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "support_cases.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS support_case (
    case_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id INTEGER NOT NULL,
    rep_id      INTEGER NOT NULL,
    reason      TEXT    NOT NULL,
    status      TEXT    NOT NULL DEFAULT 'open',
    opened_at   TEXT    NOT NULL DEFAULT (datetime('now'))
);

-- One open case per customer. Asking twice joins the existing case rather than
-- creating a duplicate for a human to deduplicate later.
CREATE UNIQUE INDEX IF NOT EXISTS one_open_case_per_customer
    ON support_case (customer_id) WHERE status = 'open';
"""


def _connect() -> sqlite3.Connection:
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    con.executescript(SCHEMA)
    return con


def open_case(customer_id: int, rep_id: int, reason: str) -> tuple[dict, bool]:
    """Open a case, or return the customer's existing open one.

    Returns (case, created). `created` is False when an open case already
    existed - the caller can then tell the customer "you're already in the
    queue" instead of silently opening a second ticket.
    """
    with _connect() as con:
        try:
            cursor = con.execute(
                "INSERT INTO support_case (customer_id, rep_id, reason) VALUES (?, ?, ?)",
                (customer_id, rep_id, reason.strip()[:500]),
            )
            row = con.execute(
                "SELECT * FROM support_case WHERE case_id = ?", (cursor.lastrowid,)
            ).fetchone()
            return dict(row), True
        except sqlite3.IntegrityError as exc:
            # Only the "one open case per customer" index may be swallowed. Any
            # other constraint failure is a real bug, and turning it into a
            # cheerful "you already have a case" would hide it.
            row = con.execute(
                "SELECT * FROM support_case WHERE customer_id = ? AND status = 'open'",
                (customer_id,),
            ).fetchone()
            if row is None:
                raise
            return dict(row), False


def open_cases_for(customer_id: int) -> list[dict]:
    with _connect() as con:
        rows = con.execute(
            "SELECT * FROM support_case WHERE customer_id = ? AND status = 'open'",
            (customer_id,),
        ).fetchall()
        return [dict(r) for r in rows]


def reset() -> None:
    """Wipe the sidecar. Tests and demo rehearsal only."""
    DB_PATH.unlink(missing_ok=True)
