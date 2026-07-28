# Chinook Records support agent

A customer support agent for a fictional music store, built with LangChain, LangGraph
and LangSmith. It ships a **deliberately broken recommendation contract (v1)** alongside
the fix (v2), because the point of the demo is not that the agent works — it's how you
find out that it doesn't.

> A correct model can produce a confidently wrong answer, and you cannot see it from the
> chat window. Reliability belongs in the tool contract, not the prompt.

## Setup

```bash
uv sync
cp .env.example .env          # add ANTHROPIC_API_KEY and LANGSMITH_API_KEY
uv run python scripts/build_db.py    # builds chinook.db, asserts every demo fixture
uv run pytest -q                     # 53 tests, no model calls, ~3s
uv run langgraph dev                 # Studio
```

## Try it

```bash
# the bug: 3 of 5 recommendations are tracks they already bought
uv run python scripts/chat.py --v1 35 "based on order 149, recommend 5 tracks"

# the fix: same question, same model, one extra WHERE predicate
uv run python scripts/chat.py 35 "based on order 149, recommend 5 tracks"

# the proof, straight from the database with no model involved
uv run python scripts/proof.py 35 149
```

## What it does

Three areas, five tools.

| Area | Tools |
|---|---|
| Account | `list_my_invoices`, `get_my_invoice` |
| Catalog | `search_catalog`, `recommend_tracks` |
| Escalation | `escalate_to_human` — the only write, gated on human approval |

Plus a back-office **case investigator** (`create_deep_agent()`) that works the queue the
support agent fills. The two are connected by that queue and nothing else — an async
boundary, not multi-agent routing.

## The two design arguments

**Customer isolation is structural.** No customer-scoped tool takes a customer id:

```python
def list_my_invoices(limit: int, runtime: ToolRuntime[Ctx]) -> ToolResult:
    customer_id = runtime.context.customer_id      # injected, never model-supplied
```

The model sees `{"limit": {"type": "integer"}}`. There is no field for someone else's
ID — not a rule it's asked to follow, not expressible. Every customer-scoped query
carries `AND CustomerId = ?`, and "doesn't exist" and "isn't yours" return the same
response so nobody can probe for other people's records.

*Scope of that claim:* it binds the agent to the identity it was handed. It is not
authentication, and the investigator's tools deliberately take an explicit customer id —
correct for a staff tool, and a deployment boundary rather than a code one.

**Guarantees live below the prompt.** v1 ranks candidates by store-wide sales and never
joins back to the customer's purchases, so it recommends things they already own — on
**225 of 412 orders (55%)**, identically across ten runs, with no model involvement. v2
adds a correlated ownership anti-join. Same model, same prompt, same tool signature.

## Layout

```
src/chinook_support/
  db.py            read-only SQLite (mode=ro), parameterized queries only
  queries.py       every SQL statement in the app, plain Python, no LangChain
  contracts.py     one result envelope for every tool: status, data, message,
                   next_actions, constraints_applied
  context.py       Ctx — the runtime identity, supplied per run
  middleware.py    human handoff (deterministic), receipts, rep injection
  cases.py         the sidecar this app writes to; Chinook stays read-only
  agent.py         create_agent(); exports graph_v1 and graph_v2
  investigator.py  create_deep_agent() case investigator
  tools/           thin adapters over queries.py
evals/             3 datasets, deterministic evaluators, paired v1/v2 experiments
scripts/           build_db · chat · proof · audit_handoffs · verify_briefs
```

SQL lives in `queries.py` rather than inside the tools so the data layer is testable
without a model or an API key, and so the recommendation query can be swapped without
touching the tool signature, prompt, or model.

## Evals

```bash
uv run python evals/datasets.py    # upload
uv run python evals/run.py         # all three, both contract versions
```

| Dataset | v1 | v2 |
|---|---|---|
| `recommendation-invariants` | 2/12 | **12/12** |
| `customer-isolation` | 10/10 | 10/10 |
| `handoff-intent` | 12/12 | 12/12 |

Isolation passing on **both** versions is the point: the recommendation bug was never a
security bug, and the suite discriminates between properties rather than going green
whenever something is fixed.

Evaluators query the database independently — one that calls the code under test can
only confirm it agrees with itself. They assert on **tool output, not the final reply**:
on one fixture the model spots the repeat and annotates it, so a judge reading the
response would score a pass while the tool underneath was returning owned tracks.

## Notes

- `data/chinook.db` is generated; `scripts/build_db.py` rebuilds and verifies it
- `uv run pytest -m adversarial` runs the 8 live-model tests (costs tokens)
- Studio: **Manage Assistants → Customer Id** switches the authenticated customer
- Restart `langgraph dev` after changing agent construction — the server is a different
  execution path from the tests, and has caught three bugs the tests could not
