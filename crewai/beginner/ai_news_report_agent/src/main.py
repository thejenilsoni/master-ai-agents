from crew import NewsAiCrew

def run():
    inputs = {'topic': 'Artificial Intelligence'}  # Change as needed
    NewsAiCrew().crew().kickoff(inputs=inputs)

if __name__ == "__main__":
    run()
