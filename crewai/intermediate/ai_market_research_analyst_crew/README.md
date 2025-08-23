# 📊 AI Market Research Analyst Crew

An AI-powered multi-agent application designed to automate the process of conducting market research, analyzing trends, and generating structured market analysis reports. This project leverages specialized AI agents to gather data from various sources, analyze market conditions, and produce comprehensive reports with actionable insights.

## ✨ Features


**🤖 Multi-Agent Architecture:**
	- � **Trends Researcher**: Finds and summarizes the latest trends for a given market or topic by analyzing news and reports to detect emerging developments.
	- 🏢 **Competitor Analyst**: Identifies key competitors, summarizes their strengths and weaknesses, and prepares SWOT analyses.
	- 🧠 **Insights Synthesizer**: Synthesizes research findings, discovers patterns, and drafts actionable recommendations for decision makers.
	- 📝 **Executive Editor**: Polishes, organizes, and formats the final market research report for clarity and professional tone.

⚙️ **Configurable Agents & Tasks**: Easily customize agent behaviors and tasks via YAML configuration files (`config/agents.yaml`, `config/tasks.yaml`).

✅ **Automated Data Collection**: Extracts and attributes key market facts and statistics from trusted sources.

📈 **Structured Output**: Generates organized market research reports saved in the `output/` directory for easy access and review.

## 🚀 Getting Started

### 1. 📥 Clone the Repository

Clone this repository and navigate to the project directory:

```powershell
git clone <your-repo-url>
cd ai_market_research_analyst_crew
```

### 2. 🛠️ Set Up the Python Environment

It is recommended to use a virtual environment:

```powershell
python -m venv crewaivenv
.\crewaivenv\Scripts\activate
```

### 3. 📦 Install Dependencies

Install the required Python packages:

```powershell
pip install -r requirements.txt
```

### 4. 🔑 Configure API Keys

Set up your API keys (e.g., for OpenAI or data APIs) in the `.env` file. Example `.env` entry for OpenAI:

```
OPENAI_API_KEY=your_openai_api_key
```

### 5. ⚙️ Configure Agents and Tasks

Edit the YAML files in the `config/` directory to customize agent roles and tasks as needed.

### 6. ▶️ Run the Application

Run the main script to start the market research process:

```powershell
python src/main.py
```

## 🖥️ Usage

- The application will automatically gather market data, analyze content, and generate a structured report saved in the `output/` directory.
- You can modify agent behaviors and tasks by editing the configuration files in `config/`.

## 📂 Project Structure

- `src/` – Source code for the agents and main application logic
- `config/` – YAML configuration files for agents and tasks
- `output/` – Generated market research reports
- `requirements.txt` – Python dependencies
- `.env` – Environment variables (API keys, etc.)
