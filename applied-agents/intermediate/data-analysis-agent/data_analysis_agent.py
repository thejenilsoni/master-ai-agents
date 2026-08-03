"""
Data Analysis Agent (Applied Agents - Intermediate)

Ask a question about a CSV in plain English and get an answer whose numbers are
real:

    python data_analysis_agent.py "Which region has the worst satisfaction and why?"

The pipeline deliberately keeps arithmetic away from the model:

1. `profile_dataset()` reads the CSV with pandas and reports what is actually
   in it — inferred kinds, null counts, ranges, cardinality. No model.
2. The model reads that profile and returns an `AnalysisPlan`: a list of steps
   drawn from a **closed vocabulary of operations** (`aggregate`, `time_trend`,
   `correlation`, …). It never writes code and never states a result.
3. `validate_step()` rejects any step that references a column that does not
   exist or applies a numeric aggregation to text. Invalid steps are dropped
   with a reason, not executed.
4. `execute_step()` computes each step with pandas. **Every number in the final
   report comes from this function.**
5. The model writes the explanation from the computed tables, and
   `unsupported_numbers()` then checks the prose: any figure that does not
   appear in the computed results is flagged in the output.

Run:
    python generate_sample_data.py          # (already committed, but reproducible)
    export OPENAI_API_KEY="sk-..."
    python data_analysis_agent.py "Which channel is hurting satisfaction?"
    python data_analysis_agent.py --csv mydata.csv "What drives revenue?"
"""

from __future__ import annotations

import math
import re
import sys
from pathlib import Path
from typing import Any, Literal, TypeVar

import pandas as pd
from pydantic import BaseModel, Field

MODEL = "gpt-4o-mini"

# Bounds. The plan is capped so one vague question cannot trigger twenty
# aggregations, and result tables are capped so a high-cardinality group-by
# cannot dump 50,000 rows into a prompt.
MAX_INPUT_ROWS = 100_000
MAX_STEPS = 5
MAX_RESULT_ROWS = 20
MAX_TOP_VALUES = 5

HERE = Path(__file__).parent
DEFAULT_CSV = HERE / "sample_data" / "orders.csv"

ColumnKind = Literal["integer", "float", "date", "boolean", "categorical", "text"]
NUMERIC_KINDS = {"integer", "float"}

Aggregation = Literal["sum", "mean", "median", "min", "max", "count", "nunique"]
NUMERIC_AGGREGATIONS = {"sum", "mean", "median"}

_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


# --------------------------------------------------------------------------- #
# 1. Profiling (pandas only — the model never sees the raw rows)
# --------------------------------------------------------------------------- #
class ColumnProfile(BaseModel):
    name: str
    kind: ColumnKind
    dtype: str
    null_count: int
    null_pct: float
    unique_count: int
    minimum: str | None = None
    maximum: str | None = None
    mean: float | None = None
    top_values: list[str] = Field(default_factory=list)


class DatasetProfile(BaseModel):
    path: str
    row_count: int
    column_count: int
    truncated: bool
    columns: list[ColumnProfile]

    def column(self, name: str) -> ColumnProfile | None:
        for column in self.columns:
            if column.name == name:
                return column
        return None

    @property
    def names(self) -> list[str]:
        return [column.name for column in self.columns]


def infer_kind(series: "pd.Series") -> ColumnKind:
    """Classify a column without trusting the CSV's dtype guess alone."""
    non_null = series.dropna()
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_numeric_dtype(series):
        if non_null.empty:
            return "float"
        # An all-whole-number float column (e.g. a nullable count) reads as int.
        if (non_null % 1 == 0).all():
            return "integer"
        return "float"
    sample = non_null.astype(str).head(50)
    if len(sample) > 0 and all(_ISO_DATE_RE.match(value) for value in sample):
        return "date"
    unique = int(non_null.nunique())
    if unique <= max(20, int(0.05 * max(len(series), 1))):
        return "categorical"
    return "text"


def load_dataframe(path: Path, max_rows: int = MAX_INPUT_ROWS) -> tuple["pd.DataFrame", bool]:
    """Read a CSV with a hard row cap. Returns (frame, truncated)."""
    frame = pd.read_csv(path, nrows=max_rows + 1)
    truncated = len(frame) > max_rows
    if truncated:
        frame = frame.head(max_rows)
    return frame, truncated


def profile_dataset(frame: "pd.DataFrame", path: str, truncated: bool = False) -> DatasetProfile:
    """Describe a dataframe: kinds, nulls, ranges, cardinality, common values."""
    row_count = int(len(frame))
    columns: list[ColumnProfile] = []

    for name in frame.columns:
        series = frame[name]
        kind = infer_kind(series)
        non_null = series.dropna()
        null_count = int(series.isna().sum())

        minimum: str | None = None
        maximum: str | None = None
        mean: float | None = None
        top_values: list[str] = []

        if not non_null.empty:
            if kind in NUMERIC_KINDS:
                minimum = _format_number(float(non_null.min()))
                maximum = _format_number(float(non_null.max()))
                mean = round(float(non_null.mean()), 4)
            elif kind == "date":
                as_text = non_null.astype(str)
                minimum = str(as_text.min())
                maximum = str(as_text.max())
            else:
                counts = non_null.astype(str).value_counts().head(MAX_TOP_VALUES)
                top_values = [f"{value} ({count})" for value, count in counts.items()]

        columns.append(
            ColumnProfile(
                name=str(name),
                kind=kind,
                dtype=str(series.dtype),
                null_count=null_count,
                null_pct=round(100.0 * null_count / row_count, 2) if row_count else 0.0,
                unique_count=int(non_null.nunique()),
                minimum=minimum,
                maximum=maximum,
                mean=mean,
                top_values=top_values,
            )
        )

    return DatasetProfile(
        path=path,
        row_count=row_count,
        column_count=len(columns),
        truncated=truncated,
        columns=columns,
    )


def _format_number(value: float) -> str:
    """Format a float the way the report shows it — 2dp, no exponent."""
    if value == int(value) and abs(value) < 1e15:
        return str(int(value))
    return f"{value:.2f}"


def render_profile(profile: DatasetProfile) -> str:
    """Render the profile as the compact table the model reasons over."""
    lines = [
        f"Dataset: {profile.path}",
        f"Rows: {profile.row_count}  Columns: {profile.column_count}"
        + ("  (TRUNCATED at the row cap)" if profile.truncated else ""),
        "",
        "| column | kind | nulls | unique | range / common values |",
        "| --- | --- | --- | --- | --- |",
    ]
    for column in profile.columns:
        if column.kind in NUMERIC_KINDS:
            detail = f"{column.minimum} .. {column.maximum} (mean {column.mean})"
        elif column.kind == "date":
            detail = f"{column.minimum} .. {column.maximum}"
        else:
            detail = ", ".join(column.top_values) or "-"
        lines.append(
            f"| {column.name} | {column.kind} | {column.null_count} "
            f"({column.null_pct}%) | {column.unique_count} | {detail} |"
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# 2. The analysis plan — a closed vocabulary, not generated code
# --------------------------------------------------------------------------- #
Operation = Literal[
    "overall",  # one aggregate over the whole table
    "aggregate",  # group by one or more columns
    "filter_aggregate",  # filter, then group by
    "time_trend",  # bucket a date column and aggregate
    "distribution",  # describe one column
    "correlation",  # pearson r between two numeric columns
]

FilterOp = Literal["none", "==", "!=", ">", "<", ">=", "<=", "contains"]
Frequency = Literal["none", "D", "W", "M"]


class AnalysisStep(BaseModel):
    """One computation. Deliberately flat so the schema stays simple."""

    operation: Operation
    title: str = Field(description="What this step answers, in a few words.")
    group_by: list[str] = Field(description="Column names to group by. Empty for overall.")
    value_column: str = Field(description="The column being aggregated, or '' when unused.")
    second_column: str = Field(description="The other column for 'correlation', else ''.")
    aggregation: Aggregation
    filter_column: str = Field(description="Column to filter on for filter_aggregate, else ''.")
    filter_op: FilterOp
    filter_value: str = Field(description="Value to compare against, else ''.")
    date_column: str = Field(description="Date column for time_trend, else ''.")
    freq: Frequency = Field(description="D/W/M bucket for time_trend, else 'none'.")
    top_n: int = Field(ge=0, le=MAX_RESULT_ROWS, description="0 means all rows, up to the cap.")


class AnalysisPlan(BaseModel):
    interpretation: str = Field(description="How you read the question, in one sentence.")
    steps: list[AnalysisStep]


class StepResult(BaseModel):
    title: str
    operation: str
    columns: list[str]
    rows: list[list[str]]
    note: str = ""

    def to_markdown(self) -> str:
        head = "| " + " | ".join(self.columns) + " |"
        rule = "| " + " | ".join("---" for _ in self.columns) + " |"
        body = ["| " + " | ".join(cell for cell in row) + " |" for row in self.rows]
        table = "\n".join([head, rule, *body])
        return f"**{self.title}**\n\n{table}" + (f"\n\n_{self.note}_" if self.note else "")


def validate_step(step: AnalysisStep, profile: DatasetProfile) -> list[str]:
    """Return the reasons this step cannot be run. Empty list means it is safe.

    This is the guard that stops a plausible-sounding plan from touching a
    column that does not exist or averaging a product name.
    """
    errors: list[str] = []
    known = set(profile.names)

    def require(column: str, label: str) -> ColumnProfile | None:
        if not column:
            errors.append(f"{label} is required for '{step.operation}'")
            return None
        if column not in known:
            errors.append(f"unknown column '{column}' for {label}")
            return None
        return profile.column(column)

    for column in step.group_by:
        if column not in known:
            errors.append(f"unknown group_by column '{column}'")

    if step.operation in {"overall", "aggregate", "filter_aggregate", "time_trend"}:
        value = require(step.value_column, "value_column")
        if value and step.aggregation in NUMERIC_AGGREGATIONS and value.kind not in NUMERIC_KINDS:
            errors.append(
                f"cannot take the {step.aggregation} of '{value.name}' ({value.kind})"
            )

    if step.operation == "aggregate" and not step.group_by:
        errors.append("aggregate needs at least one group_by column; use 'overall' instead")
    if step.operation == "aggregate" and len(step.group_by) > 2:
        errors.append("group by at most two columns")

    if step.operation == "filter_aggregate":
        column = require(step.filter_column, "filter_column")
        if step.filter_op == "none":
            errors.append("filter_aggregate needs a filter_op")
        if column and step.filter_op in {">", "<", ">=", "<="}:
            if column.kind not in NUMERIC_KINDS:
                errors.append(f"cannot compare '{column.name}' ({column.kind}) with {step.filter_op}")
            elif _as_float(step.filter_value) is None:
                errors.append(f"filter_value '{step.filter_value}' is not a number")

    if step.operation == "time_trend":
        column = require(step.date_column, "date_column")
        if column and column.kind != "date":
            errors.append(f"'{column.name}' is {column.kind}, not a date column")
        if step.freq == "none":
            errors.append("time_trend needs freq D, W or M")

    if step.operation == "distribution":
        require(step.value_column, "value_column")

    if step.operation == "correlation":
        first = require(step.value_column, "value_column")
        second = require(step.second_column, "second_column")
        for column in (first, second):
            if column and column.kind not in NUMERIC_KINDS:
                errors.append(f"correlation needs numeric columns; '{column.name}' is {column.kind}")

    return errors


def _as_float(text: str) -> float | None:
    try:
        return float(str(text).replace(",", "").strip())
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------- #
# 3. Execution — every number in the report is produced here
# --------------------------------------------------------------------------- #
def _apply_filter(frame: "pd.DataFrame", step: AnalysisStep, profile: DatasetProfile) -> "pd.DataFrame":
    column = step.filter_column
    kind = profile.column(column).kind if profile.column(column) else "text"
    series = frame[column]

    if step.filter_op == "contains":
        return frame[series.astype(str).str.contains(step.filter_value, case=False, na=False)]

    if kind in NUMERIC_KINDS:
        value: Any = _as_float(step.filter_value)
    else:
        value = step.filter_value

    if step.filter_op == "==":
        return frame[series.astype(str) == str(value)] if kind not in NUMERIC_KINDS else frame[series == value]
    if step.filter_op == "!=":
        return frame[series.astype(str) != str(value)] if kind not in NUMERIC_KINDS else frame[series != value]
    if step.filter_op == ">":
        return frame[series > value]
    if step.filter_op == "<":
        return frame[series < value]
    if step.filter_op == ">=":
        return frame[series >= value]
    if step.filter_op == "<=":
        return frame[series <= value]
    return frame


def _aggregate_series(frame: "pd.DataFrame", by: list[str] | "pd.Series", column: str, how: str):
    grouped = frame.groupby(by, dropna=False)[column]
    return getattr(grouped, how)()


def execute_step(frame: "pd.DataFrame", step: AnalysisStep, profile: DatasetProfile) -> StepResult:
    """Run one validated step with pandas and return a small, renderable table."""
    limit = step.top_n if 0 < step.top_n <= MAX_RESULT_ROWS else MAX_RESULT_ROWS

    if step.operation == "overall":
        value = getattr(frame[step.value_column].dropna(), step.aggregation)()
        return StepResult(
            title=step.title,
            operation=step.operation,
            columns=["metric", "value"],
            rows=[[f"{step.aggregation}({step.value_column})", _format_number(float(value))]],
            note=f"computed over {len(frame)} rows",
        )

    if step.operation in {"aggregate", "filter_aggregate"}:
        working = _apply_filter(frame, step, profile) if step.operation == "filter_aggregate" else frame
        series = _aggregate_series(working, step.group_by, step.value_column, step.aggregation)
        series = series.sort_values(ascending=False)
        truncated = len(series) > limit
        series = series.head(limit)

        columns = [*step.group_by, f"{step.aggregation}({step.value_column})"]
        rows: list[list[str]] = []
        for key, value in series.items():
            keys = list(key) if isinstance(key, tuple) else [key]
            rows.append([str(part) for part in keys] + [_format_number(float(value))])

        note = f"computed over {len(working)} rows"
        if step.operation == "filter_aggregate":
            note += f" (filtered {step.filter_column} {step.filter_op} {step.filter_value})"
        if truncated:
            note += f"; showing the top {limit} groups"
        return StepResult(
            title=step.title, operation=step.operation, columns=columns, rows=rows, note=note
        )

    if step.operation == "time_trend":
        dates = pd.to_datetime(frame[step.date_column], format="%Y-%m-%d", errors="coerce")
        periods = dates.dt.to_period(step.freq).astype(str)
        series = _aggregate_series(frame, periods, step.value_column, step.aggregation)
        series = series.sort_index()
        truncated = len(series) > limit
        series = series.tail(limit)  # the recent end of a trend is the useful end
        rows = [[str(period), _format_number(float(value))] for period, value in series.items()]
        note = f"computed over {int(dates.notna().sum())} dated rows"
        if truncated:
            note += f"; showing the last {limit} buckets"
        return StepResult(
            title=step.title,
            operation=step.operation,
            columns=["period", f"{step.aggregation}({step.value_column})"],
            rows=rows,
            note=note,
        )

    if step.operation == "distribution":
        column = profile.column(step.value_column)
        series = frame[step.value_column].dropna()
        if column and column.kind in NUMERIC_KINDS:
            described = series.describe()
            rows = [[str(name), _format_number(float(value))] for name, value in described.items()]
            return StepResult(
                title=step.title,
                operation=step.operation,
                columns=["statistic", "value"],
                rows=rows,
                note=f"{len(series)} non-null values",
            )
        counts = series.astype(str).value_counts().head(limit)
        total = int(counts.sum())
        rows = [
            [str(value), str(int(count)), f"{100.0 * int(count) / total:.2f}"]
            for value, count in counts.items()
        ]
        return StepResult(
            title=step.title,
            operation=step.operation,
            columns=[step.value_column, "count", "pct_of_shown"],
            rows=rows,
            note=f"{len(series)} non-null values",
        )

    # correlation
    pair = frame[[step.value_column, step.second_column]].dropna()
    correlation = float(pair[step.value_column].corr(pair[step.second_column]))
    return StepResult(
        title=step.title,
        operation=step.operation,
        columns=["pair", "pearson_r", "n"],
        rows=[
            [
                f"{step.value_column} vs {step.second_column}",
                "nan" if math.isnan(correlation) else f"{correlation:.4f}",
                str(len(pair)),
            ]
        ],
        note="pairs with a null on either side are excluded",
    )


# --------------------------------------------------------------------------- #
# 4. Checking the prose against the computed numbers
# --------------------------------------------------------------------------- #
_NUMBER_RE = re.compile(r"[-+]?\$?\d[\d,]*(?:\.\d+)?%?")


def extract_numbers(text: str) -> list[str]:
    """Pull every numeric-looking token out of prose."""
    return [match.group(0) for match in _NUMBER_RE.finditer(text)]


def _token_value(token: str) -> float | None:
    cleaned = token.replace("$", "").replace(",", "").rstrip("%")
    return _as_float(cleaned)


def collect_allowed_numbers(profile: DatasetProfile, results: list[StepResult]) -> list[float]:
    """Every number the model is allowed to repeat: computed results + profile."""
    allowed: list[float] = [float(profile.row_count), float(profile.column_count)]
    for column in profile.columns:
        allowed.append(float(column.null_count))
        allowed.append(float(column.unique_count))
        if column.mean is not None:
            allowed.append(column.mean)
        for bound in (column.minimum, column.maximum):
            value = _as_float(bound) if bound else None
            if value is not None:
                allowed.append(value)
    for result in results:
        for row in result.rows:
            for cell in row:
                value = _token_value(cell)
                if value is not None:
                    allowed.append(value)
    return allowed


def unsupported_numbers(
    text: str,
    allowed: list[float],
    tolerance_ratio: float = 0.005,
) -> list[str]:
    """Numbers in the prose that no computed result backs up.

    Small integers (0-31) and plausible years are ignored: those are ordinals,
    counts of bullet points and dates, not claims about the data. Anything else
    must match a computed value within a small relative tolerance.
    """
    flagged: list[str] = []
    for token in extract_numbers(text):
        value = _token_value(token)
        if value is None:
            continue
        if value == int(value) and 0 <= value <= 31:
            continue
        if value == int(value) and 1900 <= value <= 2100:
            continue
        tolerance = max(0.01, abs(value) * tolerance_ratio)
        if any(abs(value - candidate) <= tolerance for candidate in allowed):
            continue
        if token not in flagged:
            flagged.append(token)
    return flagged


# --------------------------------------------------------------------------- #
# 5. Model calls (imported lazily so --selftest needs no key or SDK)
# --------------------------------------------------------------------------- #
T = TypeVar("T", bound=BaseModel)


class Findings(BaseModel):
    answer: str = Field(description="A direct answer to the question, 2-4 sentences.")
    observations: list[str] = Field(description="At most four specific observations.")
    caveats: list[str] = Field(description="What the data cannot tell you. May be empty.")


PLAN_SYSTEM = (
    "You plan data analyses. You are given a profile of a table and a question. "
    "Return a short plan of at most "
    f"{MAX_STEPS} steps using only the listed operations and only the column names "
    "in the profile.\n"
    "- You may NOT compute anything yourself and you may NOT state any number.\n"
    "- Use exact column names, character for character.\n"
    "- Leave unused fields as empty strings, empty lists, 'none' or 0.\n"
    "- Prefer the smallest plan that answers the question. Add a correlation step "
    "only when the question is about a relationship between two numeric columns."
)

EXPLAIN_SYSTEM = (
    "You explain completed analyses. Every table you are given was computed with "
    "pandas; those are the only numbers that exist.\n"
    "- Quote figures exactly as they appear in the tables. Never round differently, "
    "never estimate, never compute a new number (no percentages, ratios or totals "
    "that are not in a table).\n"
    "- If the tables do not answer part of the question, say so in caveats.\n"
    "- Do not speculate about causes beyond what a correlation step actually shows."
)


def _structured_call(system: str, user: str, schema: type[T], model: str = MODEL) -> T:
    """One structured-output call, returning a validated Pydantic object."""
    import os

    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        sys.exit("Set OPENAI_API_KEY (copy .env.example to .env) or run --selftest.")

    client = OpenAI()
    parse = getattr(client.chat.completions, "parse", None)
    if parse is None:
        parse = client.beta.chat.completions.parse

    completion = parse(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        response_format=schema,
    )
    parsed = completion.choices[0].message.parsed
    if parsed is None:
        raise RuntimeError("The model returned no parsable structured output.")
    return parsed


class AnalysisRun(BaseModel):
    question: str
    interpretation: str
    profile: DatasetProfile
    results: list[StepResult]
    rejected: list[str]
    findings: Findings | None = None
    unverified_figures: list[str] = Field(default_factory=list)


def analyse(frame: "pd.DataFrame", profile: DatasetProfile, question: str) -> AnalysisRun:
    """Plan -> validate -> compute -> explain -> verify."""
    plan = _structured_call(
        PLAN_SYSTEM,
        f"{render_profile(profile)}\n\nAvailable operations: overall, aggregate, "
        f"filter_aggregate, time_trend, distribution, correlation.\n\n"
        f"Question: {question}",
        AnalysisPlan,
    )

    results: list[StepResult] = []
    rejected: list[str] = []
    for step in plan.steps[:MAX_STEPS]:
        errors = validate_step(step, profile)
        if errors:
            rejected.append(f"{step.title or step.operation}: {'; '.join(errors)}")
            continue
        try:
            results.append(execute_step(frame, step, profile))
        except Exception as exc:  # a valid-looking step can still fail on real data
            rejected.append(f"{step.title or step.operation}: execution failed ({exc})")

    run = AnalysisRun(
        question=question,
        interpretation=plan.interpretation,
        profile=profile,
        results=results,
        rejected=rejected,
    )
    if not results:
        return run

    tables = "\n\n".join(result.to_markdown() for result in results)
    run.findings = _structured_call(
        EXPLAIN_SYSTEM,
        f"Question: {question}\n\nDataset: {profile.row_count} rows.\n\n"
        f"Computed results:\n\n{tables}",
        Findings,
    )

    allowed = collect_allowed_numbers(profile, results)
    prose = " ".join([run.findings.answer, *run.findings.observations, *run.findings.caveats])
    run.unverified_figures = unsupported_numbers(prose, allowed)
    return run


def render_report(run: AnalysisRun) -> str:
    """Render the whole run, computed tables first."""
    lines = [
        f"# {run.question}",
        "",
        f"_Read as:_ {run.interpretation}",
        "",
        f"**Dataset:** `{run.profile.path}` — {run.profile.row_count} rows, "
        f"{run.profile.column_count} columns",
        "",
        "## Computed results",
        "",
    ]
    if not run.results:
        lines.append("_No step in the plan could be executed._")
    for result in run.results:
        lines += [result.to_markdown(), ""]

    if run.findings:
        lines += ["## Findings", "", run.findings.answer, ""]
        if run.findings.observations:
            lines += [f"- {observation}" for observation in run.findings.observations] + [""]
        if run.findings.caveats:
            lines += ["**Caveats**", ""] + [f"- {caveat}" for caveat in run.findings.caveats] + [""]

    if run.unverified_figures:
        lines += [
            "## ⚠️ Unverified figures",
            "",
            "These numbers appear in the explanation but not in any computed table:",
            "",
        ]
        lines += [f"- `{figure}`" for figure in run.unverified_figures]
        lines.append("")

    if run.rejected:
        lines += ["## Rejected steps", ""]
        lines += [f"- {reason}" for reason in run.rejected]
        lines.append("")

    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# 6. Entry points
# --------------------------------------------------------------------------- #
def _step(**overrides: Any) -> AnalysisStep:
    """Build a step with sensible empty defaults, for tests and examples."""
    base: dict[str, Any] = {
        "operation": "overall",
        "title": "",
        "group_by": [],
        "value_column": "",
        "second_column": "",
        "aggregation": "sum",
        "filter_column": "",
        "filter_op": "none",
        "filter_value": "",
        "date_column": "",
        "freq": "none",
        "top_n": 0,
    }
    base.update(overrides)
    return AnalysisStep.model_validate(base)


def _selftest() -> None:
    """Verify profiling, step validation, execution and the number checker.

    The expected values are computed independently with the standard-library
    `csv` module, so a pandas mistake cannot mark its own homework.
    """
    import csv
    from collections import Counter

    if not DEFAULT_CSV.exists():
        sys.exit(f"Missing {DEFAULT_CSV}. Run: python generate_sample_data.py")

    with DEFAULT_CSV.open(newline="", encoding="utf-8") as handle:
        raw_rows = list(csv.DictReader(handle))

    frame, truncated = load_dataframe(DEFAULT_CSV)
    profile = profile_dataset(frame, str(DEFAULT_CSV), truncated)

    # --- profiling matches an independent stdlib count -----------------------
    assert profile.row_count == len(raw_rows) == 420, (profile.row_count, len(raw_rows))
    assert profile.column_count == 12, profile.column_count
    assert profile.truncated is False

    kinds = {column.name: column.kind for column in profile.columns}
    assert kinds["order_date"] == "date", kinds
    assert kinds["region"] == "categorical"
    assert kinds["units"] == "integer"
    assert kinds["revenue"] == "float"
    assert kinds["order_id"] == "text", "a unique id must not be treated as a category"

    expected_missing = sum(1 for row in raw_rows if row["satisfaction"] == "")
    satisfaction = profile.column("satisfaction")
    assert satisfaction is not None
    assert satisfaction.null_count == expected_missing == 37, (satisfaction.null_count, expected_missing)
    assert abs(satisfaction.null_pct - 100 * expected_missing / len(raw_rows)) < 0.01

    expected_regions = len(Counter(row["region"] for row in raw_rows))
    assert profile.column("region").unique_count == expected_regions == 5

    expected_units_max = max(int(row["units"]) for row in raw_rows)
    assert profile.column("units").maximum == str(expected_units_max)

    expected_revenue = sum(float(row["revenue"]) for row in raw_rows)
    assert abs(profile.column("revenue").mean - expected_revenue / len(raw_rows)) < 0.01

    # --- step validation rejects everything that would silently mislead -------
    good = _step(
        operation="aggregate",
        title="Revenue by region",
        group_by=["region"],
        value_column="revenue",
        aggregation="sum",
    )
    assert validate_step(good, profile) == []

    bad_column = _step(operation="aggregate", group_by=["teritory"], value_column="revenue")
    assert any("unknown group_by column" in error for error in validate_step(bad_column, profile))

    bad_math = _step(operation="aggregate", group_by=["region"], value_column="channel", aggregation="mean")
    assert any("cannot take the mean" in error for error in validate_step(bad_math, profile))

    bad_trend = _step(operation="time_trend", date_column="region", value_column="revenue", freq="M")
    assert any("not a date column" in error for error in validate_step(bad_trend, profile))
    no_freq = _step(operation="time_trend", date_column="order_date", value_column="revenue")
    assert any("needs freq" in error for error in validate_step(no_freq, profile))

    bad_corr = _step(operation="correlation", value_column="shipping_days", second_column="channel")
    assert any("needs numeric columns" in error for error in validate_step(bad_corr, profile))

    bad_filter = _step(
        operation="filter_aggregate",
        group_by=["region"],
        value_column="revenue",
        filter_column="units",
        filter_op=">",
        filter_value="lots",
    )
    assert any("is not a number" in error for error in validate_step(bad_filter, profile))

    # --- execution matches the stdlib computation ----------------------------
    result = execute_step(frame, good, profile)
    computed = {row[0]: float(row[1]) for row in result.rows}
    expected: dict[str, float] = {}
    for row in raw_rows:
        expected[row["region"]] = expected.get(row["region"], 0.0) + float(row["revenue"])
    assert set(computed) == set(expected), (set(computed), set(expected))
    for region, value in expected.items():
        assert abs(computed[region] - value) < 0.01, (region, computed[region], value)
    ordered = [float(row[1]) for row in result.rows]
    assert ordered == sorted(ordered, reverse=True), "aggregate rows must be sorted descending"

    trend = execute_step(
        frame,
        _step(
            operation="time_trend",
            title="Monthly revenue",
            date_column="order_date",
            value_column="revenue",
            aggregation="sum",
            freq="M",
        ),
        profile,
    )
    assert [row[0] for row in trend.rows] == sorted(row[0] for row in trend.rows)
    assert len(trend.rows) == 6, trend.rows  # 2025-10 .. 2026-03
    expected_first = sum(
        float(row["revenue"]) for row in raw_rows if row["order_date"].startswith("2025-10")
    )
    assert abs(float(trend.rows[0][1]) - expected_first) < 0.01

    filtered = execute_step(
        frame,
        _step(
            operation="filter_aggregate",
            title="Enterprise revenue by channel",
            group_by=["channel"],
            value_column="revenue",
            aggregation="sum",
            filter_column="customer_segment",
            filter_op="==",
            filter_value="Enterprise",
        ),
        profile,
    )
    expected_enterprise = sum(
        float(row["revenue"]) for row in raw_rows if row["customer_segment"] == "Enterprise"
    )
    assert abs(sum(float(row[1]) for row in filtered.rows) - expected_enterprise) < 0.01

    correlation = execute_step(
        frame,
        _step(
            operation="correlation",
            title="Shipping vs satisfaction",
            value_column="shipping_days",
            second_column="satisfaction",
        ),
        profile,
    )
    r_value = float(correlation.rows[0][1])
    assert -1.0 <= r_value <= 1.0
    assert r_value < -0.3, f"the planted signal should be clearly negative, got {r_value}"
    assert int(correlation.rows[0][2]) == len(raw_rows) - expected_missing

    top = execute_step(
        frame,
        _step(
            operation="aggregate",
            title="Top categories",
            group_by=["product_category"],
            value_column="revenue",
            aggregation="sum",
            top_n=2,
        ),
        profile,
    )
    assert len(top.rows) == 2 and "top 2 groups" in top.note

    # --- the number checker --------------------------------------------------
    allowed = collect_allowed_numbers(profile, [result, trend, correlation])
    assert extract_numbers("Revenue was $1,234.50, up 7.5% across 42 regions") == [
        "$1,234.50",
        "7.5%",
        "42",
    ]
    biggest_region, biggest_value = result.rows[0][0], result.rows[0][1]
    grounded = f"{biggest_region} leads with {biggest_value} in revenue across 420 orders."
    assert unsupported_numbers(grounded, allowed) == [], unsupported_numbers(grounded, allowed)
    invented = f"{biggest_region} leads with {biggest_value}, which is 63.4% of the total."
    assert unsupported_numbers(invented, allowed) == ["63.4%"], unsupported_numbers(invented, allowed)
    assert unsupported_numbers("In 2026 the top 3 regions grew.", allowed) == []

    report = render_report(
        AnalysisRun(
            question="q",
            interpretation="i",
            profile=profile,
            results=[result],
            rejected=["bad step: unknown column 'teritory'"],
            unverified_figures=["63.4%"],
        )
    )
    assert "## Computed results" in report and "Unverified figures" in report

    print("selftest passed:")
    print(f"  profiled {profile.row_count} rows x {profile.column_count} columns; "
          f"kinds and null counts match an independent csv-module pass")
    print("  invalid steps (unknown column, non-numeric mean, bad date, bad filter) all rejected")
    print(f"  pandas results match hand-computed totals; shipping vs satisfaction r = {r_value:.3f}")
    print("  ungrounded figures in prose are detected")


def _usage() -> str:
    return (
        "Usage:\n"
        '  python data_analysis_agent.py [--csv PATH] "your question"\n'
        "  python data_analysis_agent.py --profile [--csv PATH]\n"
        "  python data_analysis_agent.py --selftest"
    )


def main() -> None:
    args = sys.argv[1:]
    if args and args[0] == "--selftest":
        _selftest()
        return

    csv_path = DEFAULT_CSV
    show_profile_only = False
    question_parts: list[str] = []

    index = 0
    while index < len(args):
        token = args[index]
        if token in {"-h", "--help"}:
            print(_usage())
            return
        if token == "--profile":
            show_profile_only = True
            index += 1
        elif token == "--csv":
            if index + 1 >= len(args):
                sys.exit(f"--csv needs a path.\n\n{_usage()}")
            csv_path = Path(args[index + 1])
            index += 2
        else:
            question_parts.append(token)
            index += 1

    if not csv_path.exists():
        sys.exit(f"No such CSV: {csv_path}")

    frame, truncated = load_dataframe(csv_path)
    profile = profile_dataset(frame, str(csv_path), truncated)

    if show_profile_only:
        print(render_profile(profile))
        return

    question = " ".join(question_parts).strip() or "What are the most important patterns in this data?"
    run = analyse(frame, profile, question)
    print()
    print(render_report(run))


if __name__ == "__main__":
    main()
