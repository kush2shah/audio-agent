"""Read-only access to the Chinook catalog.

Two things matter here:

1. The connection is opened with SQLite's `mode=ro` URI. The agent physically
   cannot write to the store's system of record, no matter what it is asked to do.
2. `query()` only takes a statement plus bound parameters. There is no code path
   anywhere in this app that concatenates model output into SQL.
"""

import sqlite3
from pathlib import Path
from typing import Any

DB_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "chinook.db"


def _connect() -> sqlite3.Connection:
    con = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    return con


def query(sql: str, params: tuple[Any, ...] = ()) -> list[dict]:
    """Run a parameterized read and return plain dicts."""
    with _connect() as con:
        return [dict(row) for row in con.execute(sql, params)]


def query_one(sql: str, params: tuple[Any, ...] = ()) -> dict | None:
    rows = query(sql, params)
    return rows[0] if rows else None
