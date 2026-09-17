# Pipeline Recovery Agent Evaluation Lab

I built a pipeline recovery agent that can inspect a failed data pipeline, diagnose the
problem, propose a repair, wait for approval, and validate the result. This project asks
the next question: how reliable is that agent when the examples are less convenient than
the demo?

The system under test is my separate
[Pipeline Recovery Agent](https://github.com/anarabiyev/pipeline-recovery-agent). This
repository runs it against 38 curated cases without modifying the original databases.

## What is tested

The cases cover healthy tables, schema drift, duplicates, missing values, multiple
simultaneous problems, unsupported data, approval decisions, tool errors, and malformed
tool output.

Each run measures:

- diagnosis accuracy;
- repair action and parameter accuracy;
- final outcome, graph path, and state;
- groundedness and repair safety with an independent LLM judge;
- behaviour after tool failures;
- latency, token use, and estimated cost.

The deterministic checks know the expected answer. The LLM judge does not: it only sees
the evidence collected by the agent and the response it produced.

## Files

```text
cases.py              the 38 cases and their DuckDB fixtures
evaluate.py           isolated agent execution, fault injection, and evaluators
run.py                experiments, reports, LangSmith, and run comparison
test_evaluation.py    free tests using fake models
subject/              the Pipeline Recovery Agent being evaluated
```

I kept the project deliberately small. The cases are plain Python data, the fixtures are
visible in the same file, and the runner writes ordinary JSON and Markdown.

## Setup

```bash
git clone --recurse-submodules <repository-url>
cd pipeline-recovery-agent-evaluation
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

On Windows, activate with `.venv\Scripts\activate`. Add your OpenAI key to `.env`.

## Run the free tests

```bash
python -m pytest -q test_evaluation.py subject/pipeline-recovery-agent/test_agent.py
```

These tests use fake models and make no API calls.

## Run a real evaluation

Start with three cases:

```bash
python run.py --limit 3 --label smoke-test
```

Then run the complete baseline:

```bash
python run.py --label baseline
```

The runner saves detailed JSON and a short Markdown report in `results/`. Use
`--no-judge` if you only want the deterministic checks.

To create the curated dataset and experiment in LangSmith:

```bash
python run.py --label baseline --langsmith
```

## Compare changes

Every result records the model, prompt hash, and Git commit of the subject agent. Run the
suite again after changing a model, prompt, or agent version, then compare the two files:

```bash
python run.py \
  --compare results/baseline.json results/candidate.json \
  --output results/comparison.md
```

Add `--fail-on-regression` to return a failing exit code when a previously passing case
starts failing or disappears from the candidate run.

The 38 cases are a starting point, not a claim of production completeness. The useful
workflow is simple: when the agent fails in a new way, turn that failure into the next
case and rerun the suite.

## References

- [LangSmith evaluation documentation](https://docs.langchain.com/langsmith/evaluation)
- [OpenAI GPT-4.1 Mini model and pricing](https://developers.openai.com/api/docs/models/gpt-4.1-mini)
