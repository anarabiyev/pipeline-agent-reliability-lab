# Pipeline Recovery Agent Evaluation Lab

I already had a working pipeline recovery agent. The more useful next question was not
“can it produce a good demo?” but “how often can I actually trust it?”

This project is my answer to that question. It runs the agent against a curated set of
pipeline incidents, checks what it diagnosed and attempted, injects failures into its
tools, and records the quality, latency, token use, and estimated cost of each run.

The system under test is my separate
[Pipeline Recovery Agent](https://github.com/anarabiyev/pipeline-recovery-agent). This
repository treats it as a subject: the evaluation code does not rewrite the agent or
silently repair its answers.

## What is evaluated

The suite currently contains 38 cases:

| Area | Cases | Examples |
| --- | ---: | --- |
| Healthy data | 3 | normal, empty, and larger tables |
| Schema drift | 6 | safe renames and ambiguous schema changes |
| Duplicates | 6 | exact and conflicting duplicate records |
| Missing values | 6 | safe fills, row removal, and unsafe cases |
| Multiple issues | 5 | combinations that should usually require review |
| Unsupported input | 5 | checks the agent does not currently understand |
| Approval and tool behaviour | 7 | rejection, pause, tool errors, and malformed data |

For every case, the code checks:

- diagnosis accuracy;
- repair-tool selection and parameters;
- final outcome;
- graph trajectory and important state fields;
- unsupported claims, relevance, and repair safety through an independent LLM judge;
- behaviour after tool exceptions or malformed tool output;
- latency, token use, and estimated API cost.

The deterministic checks use the expected answer. The LLM judge deliberately does not.
It sees only the investigation evidence and the agent response.

## Project structure

```text
cases.py             curated cases and expected behaviour
scenarios.py         small DuckDB fixtures
evaluate.py          isolated execution of one agent case
evaluators.py        deterministic, state, and trajectory checks
judge.py             reference-free LLM judge
faults.py            controlled tool failures and malformed returns
experiment.py        metrics, cost, and report generation
run_evaluation.py    command-line experiment runner
compare_results.py   model, prompt, and agent-version comparison
test_evaluate.py     local tests with fake models
subject/             the Pipeline Recovery Agent under test
```

## Setup

Clone the repository with its subject agent, then create a virtual environment:

```bash
git clone --recurse-submodules <repository-url>
cd pipeline-recovery-agent-evaluation
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

On Windows, activate the environment with `.venv\Scripts\activate`.

Add your OpenAI key to `.env`. LangSmith is optional for local experiments; add its key
only if you want the curated dataset, traces, and experiment comparison in LangSmith.

## Run the tests

The unit tests use fake models and make no API calls:

```bash
python -m pytest -q test_evaluate.py subject/pipeline-recovery-agent/test_agent.py
```

## Run an evaluation

Start with three cases to check the setup and API connection:

```bash
python run_evaluation.py --limit 3 --label smoke-test
```

Then run the full baseline:

```bash
python run_evaluation.py --label baseline
```

The command writes two files to `results/`:

- a JSON file with every case, score, error, latency, and token count;
- a short Markdown report with the main results and failures.

The default model is `gpt-4.1-mini`. A full run can make roughly two agent calls and one
judge call per ordinary case, so the three-case smoke test is worth doing first. Use
`--no-judge` when you only want the deterministic measurements.

### Run the same experiment in LangSmith

```bash
python run_evaluation.py --label baseline --langsmith
```

This creates or updates the curated dataset and runs the same code evaluators and LLM
judge as a LangSmith experiment. Concurrency stays at one because the subject agent uses
process-level model and working-directory state.

## Compare a model, prompt, or agent version

Run each candidate with a clear label:

```bash
python run_evaluation.py --model gpt-4.1-mini --label model-a
python run_evaluation.py --model <another-model> --label model-b
```

Prompt variants can be passed as plain text files without editing the subject agent:

```bash
python run_evaluation.py \
  --diagnosis-prompt prompts/diagnosis-v2.txt \
  --repair-prompt prompts/repair-v2.txt \
  --label prompt-v2
```

The report stores the model name, prompt hash, and exact Git commit of the subject agent.
Compare any two saved runs with:

```bash
python compare_results.py \
  results/baseline.json \
  results/prompt-v2.json \
  --output results/baseline-vs-prompt-v2.md
```

Add `--fail-on-regression` to return a failing exit code when a previously passing case
starts failing. That gives the project a simple regression gate without hiding the
individual cases behind one overall score.

## Notes on cost

The runner reads token usage returned by the API. It includes the current GPT-4.1 Mini
rates as a default estimate. If another model is used, set its input, cached-input, and
output rates in `.env`; otherwise the local report says that pricing is not configured.
LangSmith remains the source of truth for trace-level provider cost.

This is an offline evaluation lab, not a claim that 38 cases represent every production
failure. The useful habit is to add a new case whenever a real failure teaches the agent
something new.

## References

- [LangSmith evaluation documentation](https://docs.langchain.com/langsmith/evaluation)
- [OpenAI GPT-4.1 Mini model and pricing](https://developers.openai.com/api/docs/models/gpt-4.1-mini)
