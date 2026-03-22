# Multi-Domain Research Agent

A multi-agent research application built with OpenAI's Agents SDK and Streamlit. This application empowers users to perform in-depth research across multiple domains by utilizing specialized AI agents tailored to different fields of study.

## Features

- **Multi-Agent Architecture**:
  - **Triage Agent**: Evaluates the research query and assigns it to the appropriate domain-specific agent.
  - **Domain-Specific Agents**: Dedicated agents for various domains (e.g., Computer Science, Biology, General Research) that perform targeted research.

- **Automatic Fact Collection**: Extracts key facts from research with proper source attribution.
- **Structured Report Generation**: Produces well-organized reports including titles, abstracts, domain details, citations, and comprehensive findings.
- **Interactive UI**: Powered by Streamlit, offering a user-friendly interface for submitting research queries and reviewing results.

## How to Get Started

### 1. Clone the GitHub Repository

Clone the repository and navigate to the project directory:

```bash
git clone https://github.com/thejenilsoni/master-ai-agents.git
cd master-ai-agents/openai-agents-sdk/beginner/multi-domain-research-agent
```

### 2. Install Required Packages

Install the necessary Python packages listed in requirements.txt:

```bash
pip install -r requirements.txt
```

### 3. Set Up Your OpenAI API Key

Sign up for an OpenAI account and generate an API key at [OpenAI API Keys](https://platform.openai.com/account/api-keys). 

Set the `OPENAI_API_KEY` environment variable:

```bash
export OPENAI_API_KEY='your_openai_api_key'
```

### 4. Run the Application

Start the Streamlit application:

```bash
streamlit run multi_domain_agent.py
```

Open your web browser and go to the URL displayed in the terminal (usually __http://localhost:8501__).

## Usage

- **Enter a Research Query**
    
    - Input your research topic in the Streamlit interface's text field.
    - Optionally, choose from prewritten example queries in the sidebar for a quick start.

- **Start the Research**
    - Press the "Start Research" button to begin the process.
    - Watch the research unfold in real-time within the app.

- **View the Results**
    - After completion, review the structured report, which includes a title, abstract, domain information, citations, and detailed findings.
    - Expand the raw JSON output section for debugging or additional insights.
