# AI SQL Analyst Agent (Pydantic AI)

An **intermediate** natural-language analytics agent built with **Pydantic AI**.
Ask a question in plain English and the agent inspects the database schema,
writes a **read-only** SQL query, runs it against a seeded SQLite database, and
returns a **typed, validated** result — the answer, the exact SQL it ran, the
row count, and any assumptions it made.

This is the intermediate counterpart to the beginner
[Bank Support Agent](../../beginner/ai-bank-support-agent): same Pydantic AI
ideas (typed dependency injection + structured output), now driving a real
tool-use loop against a database.

## What it demonstrates

- **Typed dependency injection** — the SQLite connection is passed in as `deps`
  and reaches every tool through `RunContext`.
- **Tool-use loop** — `list_tables`, `describe_table`, and `run_query` let the
  model discover the schema and execute SQL instead of hallucinating columns.
- **Structured, validated output** — the agent must return an `AnalysisResult`
  (answer, `sql`, `row_count`, `assumptions`), so the result is machine-usable.
- **Defense in depth against a model writing to your DB:**
  - `is_readonly_sql()` rejects anything that isn't a single `SELECT`/`WITH`.
  - a SQLite **authorizer** on the connection hard-denies every write at the
    engine level — even a query that slips past the string check cannot mutate data.

## The dataset

A small in-memory e-commerce store is seeded on every run:

| Table | Columns |
| --- | --- |
| `customers` | id, name, country, signup_date |
| `products` | id, name, category, price |
| `orders` | id, customer_id, order_date, status |
| `order_items` | id, order_id, product_id, quantity |

Revenue for a line item is `products.price * order_items.quantity`; cancelled
orders are excluded from revenue by default.

## How to Get Started

### 1. Clone and enter the project

```bash
git clone https://github.com/thejenilsoni/master-ai-agents.git
cd master-ai-agents/pydantic-ai/intermediate/ai-sql-analyst-agent
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
# Uses a default question:
python sql_analyst_agent.py

# Or ask your own:
python sql_analyst_agent.py "Which country's customers spent the most, excluding cancelled orders?"
```

## Verify it without an API key

The database seeding and the read-only guard are plain functions with a built-in
self-test — no key required:

```bash
python sql_analyst_agent.py --selftest
# selftest passed: schema seeded, guard + authorizer enforce read-only,
# top category by revenue = Displays ($1950.00)
```

## Example output

```
Q: Which product category generated the most revenue?

Answer   : The "Displays" category generated the most revenue at $1,950.00,
           driven entirely by the Ultrawide Monitor.
SQL      : SELECT p.category, SUM(p.price * oi.quantity) AS revenue FROM ...
Rows     : 3
Assumptions:
  - Cancelled orders were excluded from revenue.
```

## Extending this project

- Point `seed_database()` at a real database file or a production replica
  (keep it read-only!).
- Add a `chart` tool that renders the returned rows.
- Stream intermediate tool calls so users can watch the agent reason.
- Add row-count / cost limits and a query timeout for large databases.
