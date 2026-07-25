"""Find conversations where a customer asked for a person and didn't get one.

    uv run python scripts/audit_handoffs.py

This is the query that would have caught the handoff bug. The failure signature is
a run that *succeeded* - no error, normal latency, healthy by every generic signal -
where the customer asked for a human and no handoff was recorded.

Nothing generic finds this. You have to know the invariant to ask for it, which is
the argument for stamping business facts onto runs in the first place.
"""

from langsmith import Client

from chinook_support.middleware import HUMAN_REQUESTED

PROJECT = "chinook-support"


def main() -> None:
    client = Client()
    runs = list(client.list_runs(project_name=PROJECT, is_root=True, limit=100))

    misses, hits = [], 0
    for run in runs:
        messages = (run.inputs or {}).get("messages") or []
        asked = any(
            HUMAN_REQUESTED.search(str(m.get("content", "")))
            for m in messages
            if isinstance(m, dict) and m.get("type") == "human"
        )
        if not asked:
            continue
        handed_off = bool((run.extra.get("metadata") or {}).get("handoff"))
        if handed_off:
            hits += 1
        else:
            text = next(
                (str(m.get("content"))[:60] for m in messages if isinstance(m, dict)), ""
            )
            misses.append((run.id, text, run.status))

    print(f"scanned {len(runs)} runs")
    print(f"  asked for a human, handed off      : {hits}")
    print(f"  asked for a human, NO handoff      : {len(misses)}")

    for run_id, text, status in misses:
        print(f"\n  MISS  status={status}  {text!r}")
        print(f"        {str(run_id)}")

    if misses:
        clean = sum(1 for _, _, status in misses if status == "success")
        print(
            f"\n{len(misses)} conversation(s) where someone asked for a person and "
            f"didn't get one."
        )
        if clean:
            print(
                f"{clean} of them completed with status=success - no error, nothing "
                "a generic health check would ever surface."
            )


if __name__ == "__main__":
    main()
