from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool
from crewai.project import CrewBase, agent, task, crew

from dotenv import load_dotenv
load_dotenv()

@CrewBase
class MarketResearchCrew:
    @agent
    def trends_researcher(self):
        return Agent(
            config=self.agents_config['trends_researcher'],
            verbose=True,
            tools=[SerperDevTool()]
        )

    @agent
    def competitor_analyst(self):
        return Agent(
            config=self.agents_config['competitor_analyst'],
            verbose=True,
            tools=[SerperDevTool()]
        )

    @agent
    def insights_synthesizer(self):
        return Agent(
            config=self.agents_config['insights_synthesizer'],
            verbose=True
        )

    @agent
    def executive_editor(self):
        return Agent(
            config=self.agents_config['executive_editor'],
            verbose=True
        )

    @task
    def trends_task(self):
        return Task(
            config=self.tasks_config['trends_task']
        )

    @task
    def competitors_task(self):
        return Task(
            config=self.tasks_config['competitors_task']
        )

    @task
    def insights_task(self):
        return Task(
            config=self.tasks_config['insights_task']
        )

    @task
    def editor_task(self):
        return Task(
            config=self.tasks_config['editor_task'],
            output_file="output/research_report.md"
        )

    @crew
    def crew(self):
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True
        )
