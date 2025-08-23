# 📢 AI News Report Agent

An AI-powered multi-agent application designed to automate the process of researching, summarizing, and generating structured news reports. This project leverages specialized AI agents to gather news from various sources, analyze content, and produce comprehensive reports with citations and key insights.

## ✨ Features

**🤖 Multi-Agent Architecture**:
	- �️‍♂️ **News Finder Agent**: Finds and collects relevant news articles from various sources.
	- ✍️ **News Writer Agent**: Summarizes and writes structured news reports based on the gathered information.

⚙️ **Configurable Agents & Tasks**: Easily customize agent behaviors and tasks via YAML configuration files (`config/agents.yaml`, `config/tasks.yaml`).

✅ **Automated Fact Collection**: Extracts and attributes key facts from news sources.

📊 **Structured Output**: Generates organized reports saved in the `output/` directory for easy access and review.

## 🚀 Getting Started

### 1. 📥 Clone the Repository

Clone this repository and navigate to the project directory:

```powershell
git clone <your-repo-url>
cd ai_news_report_agent
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

Set up your API keys (e.g., for OpenAI or news APIs) in the `.env` file. Example `.env` entry for OpenAI:

```
OPENAI_API_KEY=your_openai_api_key
```

### 5. ⚙️ Configure Agents and Tasks

Edit the YAML files in the `config/` directory to customize agent roles and tasks as needed.

### 6. ▶️ Run the Application

Run the main script to start the news report generation process:

```powershell
python src/main.py
```

## 🖥️ Usage

- The application will automatically gather news, analyze content, and generate a structured report saved in the `output/` directory.
- You can modify agent behaviors and tasks by editing the configuration files in `config/`.

## 📂 Project Structure

- `src/` – Source code for the agents and main application logic
- `config/` – YAML configuration files for agents and tasks
- `output/` – Generated news reports
- `requirements.txt` – Python dependencies
- `.env` – Environment variables (API keys, etc.)





