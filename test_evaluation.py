from pathlib import Path

import duckdb
import pytest

from cases import CASES, create_database
from evaluate import JudgeDecision, evaluate_result, judge_result, run_case
from models import Diagnosis, RepairPlan
from run import summarise


class FakeModel:
    """Return a predictable duplicate diagnosis without calling OpenAI."""

    def with_structured_output(self, schema, **kwargs):
        answer = (
            Diagnosis(
                incident_type="duplicates",
                probable_cause="The order ID appears twice.",
                evidence=["order_id 1 occurs twice"],
                confidence=1.0,
            )
            if schema is Diagnosis
            else RepairPlan(
                action="remove_duplicates",
                description="Keep the first row for each order ID.",
                parameters={"key_column": "order_id", "keep": "first"},
                risk="low",
            )
        )

        class Answer:
            def invoke(self, prompt):
                return answer

        return Answer()


class FakeJudge:
    def __init__(self, safe=True):
        self.safe = safe

    def with_structured_output(self, schema, **kwargs):
        answer = JudgeDecision(
            grounded=True,
            relevant=True,
            safe=self.safe,
            reason="The response follows the supplied evidence.",
        )

        class Answer:
            def invoke(self, prompt):
                return answer

        return Answer()


def find_case(name):
    return next(case for case in CASES if case["name"] == name)


def rows(path: Path):
    with duckdb.connect(str(path), read_only=True) as connection:
        return connection.execute("SELECT COUNT(*) FROM raw_orders").fetchone()[0]


def test_all_38_cases_are_defined():
    assert len(CASES) == 38
    assert len({case["name"] for case in CASES}) == 38


def test_every_database_scenario_can_be_created(tmp_path):
    for scenario in {case["scenario"] for case in CASES}:
        assert create_database(scenario, tmp_path / f"{scenario}.duckdb").exists()


@pytest.mark.parametrize(
    ("approved", "outcome", "recovered"),
    [(True, "recovered", True), (False, "rejected", None), (None, "paused", None)],
)
def test_approval_paths_are_isolated(tmp_path, approved, outcome, recovered):
    database = create_database("duplicates_one_exact_pair", tmp_path / f"{approved}.duckdb")
    original_rows = rows(database)

    result = run_case(database, FakeModel(), approve=approved)
    expected = find_case(
        {True: "duplicates_one_exact_pair", False: "approval_schema_rejected", None: "approval_duplicates_deferred"}[approved]
    )["expected"].copy()
    if approved is False:
        expected.update(
            incident_type="duplicates",
            action="remove_duplicates",
            parameters={"key_column": "order_id", "keep": "first"},
        )

    assert evaluate_result(result, expected)["actual"]["outcome"] == outcome
    assert result.get("recovered") is recovered
    assert rows(database) == original_rows


def test_deterministic_evaluator_shows_the_failed_check():
    result = {
        "diagnosis": {"incident_type": "missing_values"},
        "repair_plan": {
            "action": "fill_missing_values",
            "parameters": {"column": "status", "value": "pending"},
        },
        "recovered": False,
        "trajectory": [],
    }
    expected = {
        "incident_type": "missing_values",
        "action": "fill_missing_values",
        "parameters": {"column": "status", "value": "unknown"},
        "outcome": "recovered",
    }

    evaluation = evaluate_result(result, expected)

    assert evaluation["passed"] is False
    assert evaluation["checks"] == {
        "diagnosis": True,
        "action": True,
        "parameters": False,
        "outcome": False,
    }


def test_parameter_list_order_does_not_matter():
    result = {
        "diagnosis": {"incident_type": "missing_values"},
        "repair_plan": {
            "action": "remove_incomplete_rows",
            "parameters": {"columns": ["unit_price", "customer_id"]},
        },
        "recovered": True,
        "repair_result": {"success": True},
        "recovery_validation": {"passed": True},
        "trajectory": [],
    }
    expected = {
        "incident_type": "missing_values",
        "action": "remove_incomplete_rows",
        "parameters": {"columns": ["customer_id", "unit_price"]},
        "outcome": "recovered",
    }

    assert evaluate_result(result, expected)["checks"]["parameters"] is True


@pytest.mark.parametrize("name", ["tool_inspection_failure", "tool_repair_failure"])
def test_tool_errors_are_recorded(tmp_path, name):
    case = find_case(name)
    database = create_database(case["scenario"], tmp_path / f"{name}.duckdb")

    result = run_case(database, FakeModel(), fault=case["fault"])

    assert result["error"]["type"] == "RuntimeError"
    assert evaluate_result(result, case["expected"])["passed"] is True


def test_malformed_noncritical_data_can_continue(tmp_path):
    case = find_case("malformed_problem_rows")
    database = create_database(case["scenario"], tmp_path / "case.duckdb")

    result = run_case(database, FakeModel(), fault=case["fault"])

    assert result["problem_rows"] == {"unexpected_shape": True}
    assert evaluate_result(result, case["expected"])["passed"] is True


def test_malformed_critical_data_becomes_an_error(tmp_path):
    case = find_case("malformed_quality_results")
    database = create_database(case["scenario"], tmp_path / "case.duckdb")

    result = run_case(database, FakeModel(), fault=case["fault"])

    assert result["error"]["type"] == "TypeError"
    assert evaluate_result(result, case["expected"])["passed"] is True


def test_llm_judge_can_pass_or_reject_reasoning():
    result = {
        "quality_results": {"duplicate_count": 2},
        "problem_rows": {"duplicate_rows": [{"order_id": 1}, {"order_id": 1}]},
        "diagnosis": {"incident_type": "duplicates"},
        "repair_plan": {"action": "remove_duplicates", "parameters": {}},
    }

    assert judge_result(result, FakeJudge(safe=True))["passed"] is True
    assert judge_result(result, FakeJudge(safe=False))["passed"] is False


def test_llm_judge_skips_missing_reasoning():
    assert judge_result({"diagnosis": {}})["checked"] is False


def test_summary_is_calculated_from_plain_records():
    records = [record("one", True, True, 1.0), record("two", False, False, 3.0)]

    summary = summarise(records)

    assert summary["pass_rate"] == 0.5
    assert summary["unsupported_claim_rate"] == 0.5
    assert summary["average_latency_seconds"] == 2.0


def record(name, passed, grounded, latency):
    return {
        "name": name,
        "passed": passed,
        "checks": {key: passed for key in ["diagnosis", "action", "parameters", "outcome"]},
        "trajectory_passed": passed,
        "state_passed": passed,
        "judge": {"checked": True, "passed": grounded, "grounded": grounded},
        "latency_seconds": latency,
        "tokens": {"input": 0, "output": 0, "total": 0},
        "estimated_cost_usd": 0.0,
        "error": None,
    }