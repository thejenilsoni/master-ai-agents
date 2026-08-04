"""
Custom CrewAI tools for the Financial Analysis Crew.

`StockDataTool` wraps yfinance so agents can pull real market data (price,
valuation multiples, margins, and recent price action) for a ticker. Building a
custom `BaseTool` like this is the main thing that distinguishes an "advanced"
crew from one that only uses off-the-shelf search tools.

The shaping of the provider's response is kept separate from the fetch, in
`percent_change` and `build_payload` below. Those are where the mistakes that
matter live -- a mislabelled field or a sign error reaches the model as a fact
about a real company -- and keeping them out of the network call is what lets
them be checked with no market data and no CrewAI installed.
"""

from __future__ import annotations

import json
from typing import Any


def percent_change(first: float, last: float) -> float | None:
    """Percentage move between two prices, or None if there is no baseline.

    Guards against a zero or missing opening price: dividing by it would raise
    inside a tool call, which the agent sees as an unexplained tool failure.
    """
    if not first:
        return None
    return round((last - first) / first * 100, 2)


def build_payload(ticker: str, info: dict[str, Any], six_month_return: float | None) -> dict[str, Any]:
    """Shape a provider response into the JSON the agent reads.

    Fields the provider did not supply are dropped rather than sent as nulls: a
    model shown `"trailing_pe": null` will sometimes reason about it as though
    the company had a P/E of zero.
    """
    data = {
        "ticker": ticker.strip().upper(),
        "name": info.get("longName") or info.get("shortName"),
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "current_price": info.get("currentPrice") or info.get("regularMarketPrice"),
        "currency": info.get("currency"),
        "market_cap": info.get("marketCap"),
        "trailing_pe": info.get("trailingPE"),
        "forward_pe": info.get("forwardPE"),
        "price_to_book": info.get("priceToBook"),
        "profit_margin": info.get("profitMargins"),
        "revenue_growth": info.get("revenueGrowth"),
        "fifty_two_week_low": info.get("fiftyTwoWeekLow"),
        "fifty_two_week_high": info.get("fiftyTwoWeekHigh"),
        "dividend_yield": info.get("dividendYield"),
        "beta": info.get("beta"),
        "six_month_return_pct": six_month_return,
        "analyst_recommendation": info.get("recommendationKey"),
    }
    return {key: value for key, value in data.items() if value is not None}


def fetch_stock_data(ticker: str) -> str:
    """Pull market data for `ticker` and return it as JSON."""
    try:
        import yfinance as yf
    except ImportError:
        return "yfinance is not installed. Run: pip install yfinance"

    try:
        stock = yf.Ticker(ticker.strip().upper())
        info = stock.info or {}
        hist = stock.history(period="6mo")

        six_month_return = None
        if not hist.empty:
            six_month_return = percent_change(hist["Close"].iloc[0], hist["Close"].iloc[-1])

        return json.dumps(build_payload(ticker, info, six_month_return), indent=2)
    except Exception as exc:  # network / ticker errors
        return f"Could not fetch data for '{ticker}': {exc}"


def build_stock_data_tool() -> Any:
    """Construct the CrewAI tool.

    The class is defined here rather than at module scope so that importing this
    module -- and therefore checking the two functions above -- does not require
    CrewAI to be installed.
    """
    from crewai.tools import BaseTool
    from pydantic import BaseModel, Field

    class StockDataInput(BaseModel):
        ticker: str = Field(..., description="The stock ticker symbol, e.g. 'AAPL'.")

    class StockDataTool(BaseTool):
        name: str = "stock_data"
        description: str = (
            "Fetch current fundamental and market data for a public company by its "
            "ticker symbol. Returns price, market cap, P/E, margins, 52-week range, "
            "and recent performance as JSON."
        )
        args_schema: type[BaseModel] = StockDataInput

        def _run(self, ticker: str) -> str:
            return fetch_stock_data(ticker)

    return StockDataTool()
