# Data Analysis Agent

Ask a question about a CSV in plain English and get an answer whose **numbers are
real**. The whole design exists to keep arithmetic away from the model — the LLM
decides *what* to compute and explains the result, but every figure is computed
by pandas.

```bash
python data_analysis_agent.py "Which region has the worst satisfaction and why?"
```

## What it demonstrates

- **Profile first, then plan** — `profile_dataset()` reports what is actually in
  the file (inferred kinds, null counts, ranges, cardinality) with no model call.
- **A closed vocabulary of operations** — the model returns an `AnalysisPlan`
  made of allowed steps (`aggregate`, `time_trend`, `correlation`, …). It never
  writes code and never states a result.
- **Validation before execution** — `validate_step()` drops any step referencing
  a column that doesn't exist, or applying a numeric aggregation to text, *with a
  reason*, rather than executing it.
- **Numbers come from code** — `execute_step()` computes with pandas, and every
  figure in the final report originates there.
- **Hallucinated-number detection** — `unsupported_numbers()` re-reads the
  model's prose and flags any figure that doesn't appear in the computed results.

That last check is the heart of the project: it verifies the explanation against
the data instead of trusting it.

## How to Get Started

### 1. Clone and enter the project

```bash
git clone https://github.com/thejenilsoni/master-ai-agents.git
cd master-ai-agents/applied-agents/intermediate/data-analysis-agent
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set your OpenAI API key

```bash
cp .env.example .env   # then edit .env
```

### 4. Run

```bash
python generate_sample_data.py     # regenerate the sample CSV (optional)
python data_analysis_agent.py "Which channel is hurting satisfaction?"
python data_analysis_agent.py --csv mydata.csv "What drives revenue?"
```

## Verify it without an API key

```bash
python data_analysis_agent.py --selftest
```

Covers profiling, step validation, execution against known values, and the
unsupported-number detector — no key required.

## Example output

```
Dataset: sample_data/support_tickets.csv  (1,200 rows × 9 columns)
Profile: region(text, 4 unique) · satisfaction(numeric, 1.0–5.0, 12 nulls) ...

Plan:
  1. aggregate      satisfaction by region        [valid]
  2. correlation    wait_minutes ~ satisfaction   [valid]
  3. aggregate      revenue by sentiment          [dropped: no column 'sentiment']

Results:
  region    mean_satisfaction
  West                   2.91
  North                  4.12

Answer: The West region has the lowest mean satisfaction at 2.91, and wait time
correlates negatively with satisfaction (r = -0.63)...

Unsupported numbers detected: none
```

## Extending this project

- Add operations to the vocabulary (cohort retention, seasonality decomposition).
- Emit a chart per step and attach it to the report.
- Cache profiles so repeat questions skip re-profiling.
- Run the plan against a database instead of a CSV, reusing the read-only guard
  from the [SQL analyst agent](../../../pydantic-ai/intermediate/ai-sql-analyst-agent).
