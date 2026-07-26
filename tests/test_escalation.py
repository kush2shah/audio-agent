"""The write path: idempotency at the storage layer, HITL at the agent layer.

The storage tests run with no model. The HITL tests need one and are marked
`adversarial` along with the other live-model tests.
"""

import os

import pytest
from langgraph.types import Command

from chinook_support import cases, queries
from chinook_support.agent import build_agent
from chinook_support.context import Ctx
from chinook_support.tools.support import escalate_to_human

CUSTOMER = 58
FRUSTRATED = "this is really frustrating, nothing works and I've tried three times"


@pytest.fixture(autouse=True)
def clean_sidecar():
    cases.reset()
    yield
    cases.reset()


# --- storage layer, no model ------------------------------------------------


def test_a_second_case_joins_the_first():
    """A duplicate request must not create a second ticket for a human to
    deduplicate. Enforced by a UNIQUE index, not by the model remembering."""
    rep = queries.support_rep_for(CUSTOMER)
    first, created_first = cases.open_case(CUSTOMER, rep["rep_id"], "one")
    second, created_second = cases.open_case(CUSTOMER, rep["rep_id"], "two")

    assert created_first is True
    assert created_second is False
    assert first["case_id"] == second["case_id"]
    assert len(cases.open_cases_for(CUSTOMER)) == 1


def test_different_customers_get_different_cases():
    for customer_id in (35, 58):
        rep = queries.support_rep_for(customer_id)
        cases.open_case(customer_id, rep["rep_id"], "help")
    assert len(cases.open_cases_for(35)) == 1
    assert len(cases.open_cases_for(58)) == 1


def test_the_case_routes_to_the_customers_own_rep():
    rep = queries.support_rep_for(CUSTOMER)
    case, _ = cases.open_case(CUSTOMER, rep["rep_id"], "help")
    assert case["rep_id"] == rep["rep_id"]


def test_escalate_exposes_no_identity_parameter():
    props = {p.lower() for p in escalate_to_human.tool_call_schema.model_json_schema()["properties"]}
    assert not {"customer_id", "user_id", "account_id"} & props


# --- agent layer, live model ------------------------------------------------

live = pytest.mark.skipif(not os.getenv("ANTHROPIC_API_KEY"), reason="needs a model key")


@pytest.mark.adversarial
@live
def test_nothing_is_written_until_a_human_approves():
    agent = build_agent()
    config = {"configurable": {"thread_id": "test-approve"}}

    result = agent.invoke(
        {"messages": [{"role": "user", "content": FRUSTRATED}]},
        config=config,
        context=Ctx(customer_id=CUSTOMER),
    )
    assert result.get("__interrupt__"), "the write should have paused for approval"
    assert cases.open_cases_for(CUSTOMER) == [], "nothing may be written while paused"

    agent.invoke(
        Command(resume={"decisions": [{"type": "approve"}]}),
        config=config,
        context=Ctx(customer_id=CUSTOMER),
    )
    assert len(cases.open_cases_for(CUSTOMER)) == 1


@pytest.mark.adversarial
@live
def test_rejecting_writes_nothing():
    agent = build_agent()
    config = {"configurable": {"thread_id": "test-reject"}}

    result = agent.invoke(
        {"messages": [{"role": "user", "content": FRUSTRATED}]},
        config=config,
        context=Ctx(customer_id=CUSTOMER),
    )
    assert result.get("__interrupt__")

    agent.invoke(
        Command(resume={"decisions": [{"type": "reject"}]}),
        config=config,
        context=Ctx(customer_id=CUSTOMER),
    )
    assert cases.open_cases_for(CUSTOMER) == [], "a rejected write must not happen"
