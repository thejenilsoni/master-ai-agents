# AI Financial Analysis Crew (CrewAI)

An **advanced** CrewAI project that runs a six-agent equity-research workflow and
produces a polished investment brief for any public company. It goes beyond the
beginner/intermediate crews by adding a **custom CrewAI tool** (`StockDataTool`)
that pulls live market data from yfinance, plus reasoning-enabled specialist
agents and a multi-stage, context-passing pipeline.

## The crew

| Agent | Tool(s) | Role |
| --- | --- | --- |
| Market Data Analyst | `stock_data` (custom) | Pulls live price, valuation, margins, performance |
| News & Sentiment Analyst | `SerperDevTool` | Finds recent catalysts, judges sentiment |
| Fundamental Analyst | — (reasoning) | Valuation, growth, profitability assessment |
| Risk Analyst | — (reasoning) | Ranks the key investment risks |
| Investment Advisor | — (reasoning) | Issues a Buy / Hold / Sell call with a thesis |
| Equity Research Editor | — | Assembles the final brief |

The tasks run sequentially, each receiving the relevant earlier tasks as
`context`, so the recommendation is grounded in the data, news, fundamentals, and
risk work that came before it.

```
market data ─> news/sentiment ─> fundamentals ─> risk ─> recommendation ─> editor
     │ stock_data        │ SerperDev                                    │
     └──────────── live data + sources flow forward as context ────────┘
```

## What makes it "advanced"

- A **custom `BaseTool`** (`src/tools.py`) that fetches real data from yfinance.
- **Reasoning** enabled on the analytical agents (`reasoning: true`).
- A longer pipeline with explicit `context` dependencies between tasks.
- A real, shareable artifact written to `output/equity_research_brief.md`.

## How to Get Started

### 1. Clone and enter the project

```bash
git clone https://github.com/thejenilsoni/master-ai-agents.git
cd master-ai-agents/crewai/advanced/ai_financial_analysis_crew
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure your API keys

```bash
cp .env.example .env   # then edit it
```

- `OPENAI_API_KEY` — for the agents' LLM.
- `SERPER_API_KEY` — for web/news search ([serper.dev](https://serper.dev)).

### 4. Run the crew

```bash
cd src
python main.py
```

By default it analyzes **NVDA**. Edit the `inputs` in `src/main.py` to point at a
different ticker/company. The final brief is written to `output/equity_research_brief.md`.

## Verifying it

```bash
cd src && python main.py --selftest    # 34 checks, no API key, no market data
```

Two things are checked, both of which fail expensively otherwise:

**The configuration.** Agents and tasks are defined in YAML and referenced from
`crew.py` by string key, so a renamed or mistyped key is a `KeyError` that only
appears once `kickoff()` has started calling a paid model. The self-test reads
both files and checks they agree — including that every task is assigned to an
agent that exists.

**The stock tool.** Response shaping lives in `percent_change` and
`build_payload`, outside the network call, so it can be checked with no market
data. A zero opening price yields no figure rather than dividing by zero inside
a tool call, and fields the provider did not return are dropped rather than sent
as nulls — a model shown `"trailing_pe": null` will sometimes reason about it as
a P/E of zero.

## ⚠️ Disclaimer

This project is for educational and demonstration purposes only. Its output is
**not financial advice**. Always do your own research and consult a licensed
professional before investing.

## Extending this project

- Add a `Technical Analyst` agent with a custom indicators tool.
- Swap the sequential process for `Process.hierarchical` with a manager agent.
- Have the crew compare two tickers and recommend the better risk-adjusted buy.
