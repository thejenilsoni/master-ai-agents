from crew import FinancialAnalysisCrew


def run():
    # Change the ticker/company to analyze a different stock.
    inputs = {
        "ticker": "NVDA",
        "company": "NVIDIA Corporation",
    }
    FinancialAnalysisCrew().crew().kickoff(inputs=inputs)


if __name__ == "__main__":
    run()
