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


def handoff_traces(client: Client) -> set:
    """Trace ids where a handoff actually fired.

    The handoff receipt is written from inside a middleware hook, which means it
    lands on that hook's *span*, not on the root run - `parent_run` is None on a
    reconstructed tree, so there's no walking up, and patching the root
    afterwards 409s.

    An earlier version of this script read `handoff` off the root run's metadata,
    where it never appears. It reported every handoff as a miss, and looked
    exactly the same as a genuinely broken handoff. Query the spans instead.
    """
    spans = client.list_runs(
        project_name=PROJECT, filter='eq(metadata_key, "handoff_reason")', limit=100
    )
    return {span.trace_id for span in spans}


def main() -> None:
    client = Client()
    runs = list(client.list_runs(project_name=PROJECT, is_root=True, limit=100))
    handed_off_traces = handoff_traces(client)

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
        if run.trace_id in handed_off_traces:
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
