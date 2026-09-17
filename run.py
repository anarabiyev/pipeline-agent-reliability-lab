"""Run an evaluation or compare two saved runs."""

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.callbacks import UsageMetadataCallbackHandler
from langchain_openai import ChatOpenAI

from cases import CASES, create_database
from evaluate import AGENT_FOLDER, agent, evaluate_result, judge_result, run_case


ROOT = Path(__file__).parent


def main():
    load_dotenv()
    args = arguments()
    if args.compare:
        compare_runs(*args.compare, args.output, args.fail_on_regression)
        return

    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("Add OPENAI_API_KEY to .env first.")

    cases = select_cases(args)
    if args.langsmith:
        if not os.getenv("LANGSMITH_API_KEY"):
            raise SystemExit("Add LANGSMITH_API_KEY to .env or remove --langsmith.")
        records, langsmith = run_in_langsmith(cases, args)
    else:
        records = [run_record(case, args, number, len(cases)) for number, case in enumerate(cases, 1)]
        langsmith = None

    document = {
        "run": run_info(args, len(cases), langsmith),
        "summary": summarise(records),
        "cases": records,
    }
    path = args.output or ROOT / "results" / f"{safe_name(args.label)}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2), encoding="utf-8")
    report = path.with_suffix(".md")
    report.write_text(report_text(document), encoding="utf-8")

    print(f"\nPass rate: {percent(document['summary']['pass_rate'])}")
    print(f"Saved: {path}")
    print(f"Report: {report}")


def arguments():
    parser = argparse.ArgumentParser(description="Evaluate the Pipeline Recovery Agent.")
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"))
    parser.add_argument("--judge-model", default=os.getenv("OPENAI_JUDGE_MODEL", "gpt-4.1-mini"))
    parser.add_argument("--label", default="baseline")
    parser.add_argument("--case", action="append")
    parser.add_argument("--category", action="append")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--no-judge", action="store_true")
    parser.add_argument("--langsmith", action="store_true")
    parser.add_argument("--dataset", default="pipeline-recovery-agent-evaluation")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--compare", nargs=2, type=Path, metavar=("BASELINE", "CANDIDATE"))
    parser.add_argument("--fail-on-regression", action="store_true")
    return parser.parse_args()


def select_cases(args):
    selected = CASES
    if args.case:
        unknown = set(args.case) - {case["name"] for case in CASES}
        if unknown:
            raise SystemExit(f"Unknown case: {', '.join(sorted(unknown))}")
        selected = [case for case in selected if case["name"] in args.case]
    if args.category:
        selected = [case for case in selected if case["category"] in args.category]
    if args.limit:
        selected = selected[: args.limit]
    if not selected:
        raise SystemExit("No cases matched.")
    return selected


def run_record(case, args, number=None, total=None):
    if number:
        print(f"[{number}/{total}] {case['name']}")

    agent_usage = UsageMetadataCallbackHandler()
    model = ChatOpenAI(model=args.model, temperature=0, callbacks=[agent_usage])
    with tempfile.TemporaryDirectory() as folder:
        database = create_database(case["scenario"], Path(folder) / "case.duckdb")
        started = time.perf_counter()
        result = run_case(database, model, case["approve"], case.get("fault"))
        evaluation = evaluate_result(result, case["expected"])

        judge_usage = UsageMetadataCallbackHandler()
        if args.no_judge:
            judgment = {"checked": False, "passed": None, "reason": "Judge disabled."}
        else:
            judge_model = ChatOpenAI(
                model=args.judge_model,
                temperature=0,
                callbacks=[judge_usage],
            )
            judgment = judge_result(result, judge_model)
        latency = time.perf_counter() - started

    agent_tokens = token_usage(agent_usage)
    judge_tokens = token_usage(judge_usage)
    agent_cost = estimated_cost(args.model, agent_tokens)
    judge_cost = estimated_cost(args.judge_model, judge_tokens, "JUDGE_")
    costs = [cost for cost, tokens in [(agent_cost, agent_tokens), (judge_cost, judge_tokens)] if tokens["total"]]

    return {
        "name": case["name"],
        "category": case["category"],
        "expected": case["expected"],
        "actual": evaluation["actual"],
        "checks": evaluation["checks"],
        "passed": evaluation["passed"],
        "trajectory_passed": evaluation["trajectory_passed"],
        "state_passed": evaluation["state_passed"],
        "judge": judgment,
        "latency_seconds": round(latency, 3),
        "tokens": add_tokens(agent_tokens, judge_tokens),
        "estimated_cost_usd": None if any(cost is None for cost in costs) else round(sum(costs), 8),
        "error": result.get("error"),
        "agent_output": {
            key: result[key]
            for key in ["diagnosis", "repair_plan", "repair_result", "recovered", "trajectory"]
            if key in result
        },
    }


def summarise(records):
    checked_judges = [record for record in records if record["judge"]["checked"]]
    latencies = sorted(record["latency_seconds"] for record in records)
    costs = [record["estimated_cost_usd"] for record in records]

    def rate(values):
        values = list(values)
        return round(sum(bool(value) for value in values) / len(values), 4) if values else None

    return {
        "cases": len(records),
        "pass_rate": rate(record["passed"] for record in records),
        "diagnosis_accuracy": rate(record["checks"]["diagnosis"] for record in records),
        "tool_selection_accuracy": rate(record["checks"]["action"] for record in records),
        "tool_parameter_accuracy": rate(record["checks"]["parameters"] for record in records),
        "outcome_accuracy": rate(record["checks"]["outcome"] for record in records),
        "trajectory_pass_rate": rate(record["trajectory_passed"] for record in records if record["trajectory_passed"] is not None),
        "state_pass_rate": rate(record["state_passed"] for record in records if record["state_passed"] is not None),
        "judge_pass_rate": rate(record["judge"]["passed"] for record in checked_judges),
        "unsupported_claim_rate": rate(not record["judge"].get("grounded", True) for record in checked_judges),
        "errors": sum(bool(record["error"]) for record in records),
        "average_latency_seconds": round(sum(latencies) / len(latencies), 3),
        "p95_latency_seconds": latencies[max(0, math.ceil(len(latencies) * 0.95) - 1)],
        "tokens": sum_tokens(record["tokens"] for record in records),
        "estimated_cost_usd": None if any(cost is None for cost in costs) else round(sum(costs), 6),
    }


def run_info(args, case_count, langsmith):
    prompts = f"{agent.DIAGNOSIS_PROMPT}\n{agent.REPAIR_PROMPT}".encode()
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=AGENT_FOLDER,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    return {
        "label": args.label,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "judge_model": args.judge_model,
        "case_count": case_count,
        "prompt_hash": hashlib.sha256(prompts).hexdigest()[:12],
        "agent_commit": commit,
        "langsmith": langsmith,
    }


def report_text(document):
    run, summary = document["run"], document["summary"]
    lines = [
        f"# Evaluation report: {run['label']}",
        "",
        f"Model: `{run['model']}`  ",
        f"Cases: {summary['cases']}  ",
        f"Agent commit: `{run['agent_commit']}`  ",
        f"Prompt hash: `{run['prompt_hash']}`",
        "",
        "| Measure | Result |",
        "| --- | ---: |",
    ]
    for key, label in [
        ("diagnosis_accuracy", "Diagnosis accuracy"),
        ("tool_selection_accuracy", "Tool selection accuracy"),
        ("tool_parameter_accuracy", "Tool parameter accuracy"),
        ("outcome_accuracy", "Outcome accuracy"),
        ("judge_pass_rate", "LLM-judge pass rate"),
        ("unsupported_claim_rate", "Unsupported claim rate"),
    ]:
        lines.append(f"| {label} | {percent(summary[key])} |")
    lines.extend([
        f"| Average latency | {summary['average_latency_seconds']:.2f}s |",
        f"| P95 latency | {summary['p95_latency_seconds']:.2f}s |",
        f"| Estimated cost | {money(summary['estimated_cost_usd'])} |",
        "",
        "## Cases needing attention",
        "",
    ])
    attention = [
        record for record in document["cases"]
        if not record["passed"] or (record["judge"]["checked"] and not record["judge"]["passed"])
    ]
    lines.extend(
        f"- `{record['name']}`: "
        + ", ".join(name for name, passed in record["checks"].items() if not passed)
        for record in attention
    )
    if not attention:
        lines.append("None.")
    return "\n".join(lines) + "\n"


def run_in_langsmith(cases, args):
    from langsmith import Client

    client = Client()
    examples = sync_dataset(client, args.dataset, cases)

    def target(inputs):
        selected = next(case for case in cases if case["name"] == inputs["name"])
        return {"record": run_record(selected, args)}

    def deterministic(outputs, reference_outputs):
        return {"key": "deterministic", "score": outputs["record"]["passed"]}

    def grounded(outputs, reference_outputs):
        return {"key": "grounded", "score": outputs["record"]["judge"].get("grounded")}

    results = client.evaluate(
        target,
        data=examples,
        evaluators=[deterministic, grounded],
        experiment_prefix=args.label,
        max_concurrency=1,
        metadata={"model": args.model, "judge_model": args.judge_model},
    )
    rows = list(results)
    by_name = {row["run"].outputs["record"]["name"]: row["run"].outputs["record"] for row in rows}
    return [by_name[case["name"]] for case in cases], {
        "experiment": results.experiment_name,
        "url": results.url,
        "dataset": args.dataset,
    }


def sync_dataset(client, name, cases):
    if not client.has_dataset(dataset_name=name):
        client.create_dataset(name, description="Pipeline Recovery Agent reliability cases.")
    existing = {example.inputs["name"]: example for example in client.list_examples(dataset_name=name)}
    examples = []
    for case in cases:
        inputs = {key: case.get(key) for key in ["name", "category", "scenario", "approve", "fault"]}
        example = existing.get(case["name"])
        if example is None:
            example = client.create_example(dataset_name=name, inputs=inputs, outputs=case["expected"])
        elif example.inputs != inputs or example.outputs != case["expected"]:
            client.update_example(example.id, inputs=inputs, outputs=case["expected"])
            example = next(client.list_examples(dataset_name=name, example_ids=[example.id]))
        examples.append(example)
    return examples


def compare_runs(baseline_path, candidate_path, output, fail_on_regression):
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    before = {case["name"]: case for case in baseline["cases"]}
    after = {case["name"]: case for case in candidate["cases"]}
    shared = sorted(set(before) & set(after))
    regressions = [
        name for name in shared
        if (before[name]["passed"] and not after[name]["passed"])
        or (
            before[name]["judge"]["checked"]
            and before[name]["judge"]["passed"]
            and after[name]["judge"]["checked"]
            and not after[name]["judge"]["passed"]
        )
    ]
    missing = sorted(set(before) - set(after))
    text = (
        f"# {baseline['run']['label']} vs {candidate['run']['label']}\n\n"
        f"Baseline pass rate: {percent(baseline['summary']['pass_rate'])}  \n"
        f"Candidate pass rate: {percent(candidate['summary']['pass_rate'])}  \n"
        f"Regressions: {', '.join(regressions) or 'none'}  \n"
        f"Missing cases: {', '.join(missing) or 'none'}\n"
    )
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
        print(f"Saved: {output}")
    else:
        print(text)
    if fail_on_regression and (regressions or missing):
        raise SystemExit(1)


def token_usage(handler):
    total = {"input": 0, "output": 0, "total": 0}
    for usage in handler.usage_metadata.values():
        total["input"] += usage.get("input_tokens", 0)
        total["output"] += usage.get("output_tokens", 0)
        total["total"] += usage.get("total_tokens", 0)
    return total


def estimated_cost(model, tokens, prefix=""):
    defaults = (0.40, 1.60) if model.startswith("gpt-4.1-mini") else (None, None)
    input_rate = os.getenv(f"{prefix}INPUT_COST_PER_MILLION", defaults[0])
    output_rate = os.getenv(f"{prefix}OUTPUT_COST_PER_MILLION", defaults[1])
    if input_rate is None or output_rate is None:
        return None
    return (tokens["input"] * float(input_rate) + tokens["output"] * float(output_rate)) / 1_000_000


def add_tokens(first, second):
    return {key: first[key] + second[key] for key in first}


def sum_tokens(items):
    total = {"input": 0, "output": 0, "total": 0}
    for item in items:
        total = add_tokens(total, item)
    return total


def percent(value):
    return "not checked" if value is None else f"{value * 100:.1f}%"


def money(value):
    return "rate not configured" if value is None else f"${value:.4f}"


def safe_name(value):
    return re.sub(r"[^a-zA-Z0-9._-]+", "-", value).strip("-") or "experiment"


if __name__ == "__main__":
    main()
