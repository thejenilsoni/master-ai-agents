.PHONY: install run app test lint typecheck check demo

install:
	python -m pip install -e ".[dev]"

run:
	deep-research "$(QUERY)"

app:
	streamlit run app/streamlit_app.py

test:
	pytest --cov=deep_research_agent --cov-report=term-missing

lint:
	ruff check .
	ruff format --check .

typecheck:
	mypy src

check: lint typecheck test

demo:
	deep-research "What changed in production AI-agent engineering over the last 12 months, and which patterns are durable?" --auto-approve
