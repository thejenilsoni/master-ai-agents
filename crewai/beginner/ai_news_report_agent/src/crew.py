from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import SerperDevTool

from dotenv import load_dotenv
load_dotenv()

@CrewBase
class NewsAiCrew:
    @agent
    def news_finder(self):
        return Agent(
            config=self.agents_config['news_finder'],
            verbose=True,
            tools=[SerperDevTool()]
        )

    @agent
    def news_writer(self):
        return Agent(
            config=self.agents_config['news_writer'],
            verbose=True
        )

    @task
    def find_news_task(self):
        return Task(
            config=self.tasks_config['find_news_task']
        )

    @task
    def write_report_task(self):
        return Task(
            config=self.tasks_config['write_report_task'],
            output_file="output/report.md"
        )

    @crew
    def crew(self):
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,  # Finder → Writer
            verbose=True
        )
