"""Run the experiments.

    uv run python evals/datasets.py          # once, to upload
    uv run python evals/run.py               # all three datasets, v1 and v2
    uv run python evals/run.py recommendation-invariants

The target runs the real agent and then extracts the structured facts the
evaluators need - which track ids the tools returned, which invoices, whether a
handoff happened. Studio stays conversational; the evaluators stay deterministic.
"""

import json
import sys
import uuid
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parent))

from langsmith import Client, evaluate  # noqa: E402

from chinook_support.agent import build_agent  # noqa: E402
from chinook_support.context import Ctx  # noqa: E402
import evaluators as ev  # noqa: E402

DATASETS = {
    "recommendation-invariants": [
        ev.no_recommended_track_is_owned,
        ev.recommendations_exist_in_catalog,
        ev.tool_calls_correctly_scoped,
    ],
    "customer-isolation": [
        ev.no_cross_customer_leak,
        ev.tool_calls_correctly_scoped,
    ],
    "handoff-intent": [
        ev.handed_off_when_asked,
        ev.rep_details_are_real,
    ],
}


def _text(message) -> str:
    content = message.content
    if isinstance(content, list):
        return " ".join(
            part.get("text", "") for part in content if isinstance(part, dict)
        )
    return str(content)


def make_target(version: str):
    """Build a target that runs the agent and reports what the tools did."""
    agent = build_agent(version)

    def target(inputs: dict) -> dict:
        customer_id = inputs["customer_id"]
        result = agent.invoke(
            {"messages": [{"role": "user", "content": inputs["question"]}]},
            config={"configurable": {"thread_id": str(uuid.uuid4())}},
            context=Ctx(customer_id=customer_id),
        )

        track_ids: list[int] = []
        invoice_ids: list[int] = []
        statuses: list[str] = []
        called_escalate = False

        for message in result["messages"]:
            for call in getattr(message, "tool_calls", None) or []:
                if call["name"] == "escalate_to_human":
                    called_escalate = True
            if getattr(message, "type", None) != "tool":
                continue
            try:
                payload = json.loads(message.content)
            except (json.JSONDecodeError, TypeError):
                continue
            statuses.append(payload.get("status", "unknown"))
            for row in payload.get("data") or []:
                if "track_id" in row and "store_sales" in row:
                    track_ids.append(row["track_id"])  # a recommendation, not a lookup
                if "invoice_id" in row:
                    invoice_ids.append(row["invoice_id"])

        answer = _text(result["messages"][-1])
        # The deterministic handoff ends the turn itself, so a handoff shows up
        # either as the escalation tool being called or as that closing message.
        handed_off = called_escalate or "case #" in answer.lower()

        return {
            "customer_id": customer_id,
            "answer": answer,
            "recommended_track_ids": track_ids,
            "invoice_ids_returned": invoice_ids,
            "tool_statuses": statuses,
            "handed_off": handed_off,
        }

    return target


def main() -> None:
    wanted = sys.argv[1:] or list(DATASETS)
    client = Client()

    for name in wanted:
        if name not in DATASETS:
            sys.exit(f"unknown dataset {name!r}. one of: {', '.join(DATASETS)}")
        for version in ("v1", "v2"):
            print(f"\n=== {name} / {version} ===")
            evaluate(
                make_target(version),
                data=name,
                evaluators=DATASETS[name],
                experiment_prefix=f"{name}-{version}",
                metadata={"contract_version": version},
                max_concurrency=4,
                client=client,
            )


if __name__ == "__main__":
    main()
