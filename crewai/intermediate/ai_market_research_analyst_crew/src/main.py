from crew import MarketResearchCrew

def run():
    inputs = {'topic': 'Generative AI startups in Recruitment Industry'}  # or any topic you want!
    MarketResearchCrew().crew().kickoff(inputs=inputs)

if __name__ == "__main__":
    run()
