from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import SerperDevTool

from dotenv import load_dotenv

from tools import build_stock_data_tool

load_dotenv()


@CrewBase
class FinancialAnalysisCrew:
    """A multi-specialist crew that produces an equity research brief.

    Pipeline (sequential):
        market data -> news sentiment -> fundamentals -> risk -> recommendation -> editor
    """

    @agent
    def market_data_analyst(self):
        return Agent(
            config=self.agents_config["market_data_analyst"],
            verbose=True,
            tools=[build_stock_data_tool()],
        )

    @agent
    def news_sentiment_analyst(self):
        return Agent(
            config=self.agents_config["news_sentiment_analyst"],
            verbose=True,
            tools=[SerperDevTool()],
        )

    @agent
    def fundamental_analyst(self):
        return Agent(
            config=self.agents_config["fundamental_analyst"],
            verbose=True,
        )

    @agent
    def risk_analyst(self):
        return Agent(
            config=self.agents_config["risk_analyst"],
            verbose=True,
        )

    @agent
    def investment_advisor(self):
        return Agent(
            config=self.agents_config["investment_advisor"],
            verbose=True,
        )

    @agent
    def report_editor(self):
        return Agent(
            config=self.agents_config["report_editor"],
            verbose=True,
        )

    @task
    def market_data_task(self):
        return Task(config=self.tasks_config["market_data_task"])

    @task
    def sentiment_task(self):
        return Task(config=self.tasks_config["sentiment_task"])

    @task
    def fundamentals_task(self):
        return Task(config=self.tasks_config["fundamentals_task"])

    @task
    def risk_task(self):
        return Task(config=self.tasks_config["risk_task"])

    @task
    def recommendation_task(self):
        return Task(config=self.tasks_config["recommendation_task"])

    @task
    def editor_task(self):
        return Task(
            config=self.tasks_config["editor_task"],
            output_file="output/equity_research_brief.md",
        )

    @crew
    def crew(self):
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
