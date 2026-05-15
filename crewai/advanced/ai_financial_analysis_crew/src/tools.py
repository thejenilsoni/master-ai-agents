"""
Custom CrewAI tools for the Financial Analysis Crew.

`StockDataTool` wraps yfinance so agents can pull real market data (price,
valuation multiples, margins, and recent price action) for a ticker. Building a
custom `BaseTool` like this is the main thing that distinguishes an "advanced"
crew from one that only uses off-the-shelf search tools.
"""

from __future__ import annotations

import json

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
        try:
            import yfinance as yf
        except ImportError:
            return "yfinance is not installed. Run: pip install yfinance"

        try:
            stock = yf.Ticker(ticker.strip().upper())
            info = stock.info or {}
            hist = stock.history(period="6mo")

            perf_6mo = None
            if not hist.empty:
                first, last = hist["Close"].iloc[0], hist["Close"].iloc[-1]
                if first:
                    perf_6mo = round((last - first) / first * 100, 2)

            data = {
                "ticker": ticker.upper(),
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
                "six_month_return_pct": perf_6mo,
                "analyst_recommendation": info.get("recommendationKey"),
            }
            # Drop empty fields so the model isn't confused by nulls.
            data = {k: v for k, v in data.items() if v is not None}
            return json.dumps(data, indent=2)
        except Exception as exc:  # network / ticker errors
            return f"Could not fetch data for '{ticker}': {exc}"
