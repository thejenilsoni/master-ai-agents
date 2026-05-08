# Startup Idea Validator System

An **advanced** multi-agent application built with the **OpenAI Agents SDK** and
Streamlit. It assembles a panel of AI advisors that stress-test a startup idea —
the way a seasoned investor or accelerator would — and returns a **scored,
structured verdict**: Go, Pivot, or No-Go.

## Architecture

This project showcases two advanced Agents SDK patterns working together:

### 1. Agents-as-tools (the manager pattern)

A **Validation Director** agent orchestrates five specialist agents, each
exposed to it as a callable tool:

| Specialist | What it analyzes |
| --- | --- |
| 📈 Market Analyst | Market size, demand signals, growth, timing (with web search) |
| 🥊 Competitor Analyst | Direct/indirect competitors and possible moat (with web search) |
| 🧑‍🤝‍🧑 Customer Validator | Target segment, core problem, painkiller vs. vitamin |
| 💰 Business Model Analyst | Monetization, unit economics, go-to-market |
| ⚠️ Risk Analyst | The biggest reasons the idea could fail |

### 2. Structured final output

A **Synthesizer** agent reads the director's dossier and emits a typed
`ValidationReport` Pydantic model via the SDK's `output_type`, so the UI renders
clean scores and lists instead of free-form text.

```
idea ──> Validation Director ──┬─ analyze_market   (Market Analyst + WebSearch)
                               ├─ analyze_competition (Competitor Analyst + WebSearch)
                               ├─ validate_customer
                               ├─ analyze_business_model
                               └─ assess_risks
                                        │  dossier
                                        ▼
                                  Synthesizer (output_type=ValidationReport)
                                        │
                                        ▼
                          { scores, verdict, strengths, risks, next steps }
```

## How to Get Started

### 1. Clone and enter the project

```bash
git clone https://github.com/thejenilsoni/master-ai-agents.git
cd master-ai-agents/openai-agents-sdk/advanced/startup-idea-validator-system
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add your OpenAI API key

Either export it:

```bash
export OPENAI_API_KEY="sk-..."
```

…or copy the secrets template:

```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml   # then edit it
```

### 4. Run the app

```bash
streamlit run startup_idea_validator.py
```

Open the URL it prints (usually http://localhost:8501), describe your idea (or
pick an example from the sidebar), and click **Validate idea**.

## Notes

- The specialists that use `WebSearchTool` call OpenAI's hosted web search, so
  results reflect current information when available.
- A full run consults five specialists and then synthesizes — expect it to take
  up to a minute and to use a fair number of tokens.

## Extending this project

- Add a `Financial Modeler` specialist that projects a simple 3-year P&L.
- Persist reports to a database and add a comparison view across ideas.
- Swap `gpt-4o` for a cheaper model on the specialists to cut cost.
