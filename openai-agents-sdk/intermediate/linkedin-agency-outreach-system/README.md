# 👥 LinkedIn Agency Outreach System

This project leverages OpenAI's Agent SDK, Streamlit, and various tools to automate LinkedIn outreach for premium agency services. The system orchestrates agents to research leads, extract LinkedIn profile data, and generate personalized outreach emails.

## 🛠️ Features
- **Lead Research Specialist**: Scrapes and parses LinkedIn profiles using the Proxycurl API.
- **Cold Email Outreach Specialist**: Drafts highly personalized emails tailored to the recipient's profile.
- **Agency Team Lead**: Manages the workflow, ensures quality, and consolidates outputs.
- **Streamlit Interface**: Simplifies input and displays results for users.


## 🚀 How to Set Up and Run

### 1. Clone the Repository
```bash
git clone https://github.com/thejenilsoni/master-ai-agents.git
cd master-ai-agents/openai-agents-sdk/intermediate/linkedin-agency-outreach-system/
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure Secrets
Get your [OPENAI_API_KEY](https://platform.openai.com/account/api-keys) and [PROXYCURL_API_KEY](https://nubela.co/proxycurl/)

Add your credentials in `.streamlit/secrets.toml` file 
```bash
OPENAI_API_KEY = "your_openai_api_key_here"
PROXYCURL_API_KEY = "your_proxycurl_api_key_here"
```

### 4. Run the Application
Start the Streamlit application:
```bash
streamlit run linkedin_agency_outreach_system.py
```
Open your web browser and go to the URL displayed in the terminal (usually __http://localhost:8501__).


## 🖥️ Usage

### 1. Enter:
- Name
- LinkedIn profile URL

### 2. Click Generate Outreach Email. The system will:
- Scrape and parse LinkedIn profile data.
- Draft a personalized email.
- Display the results, including the generated email and raw profile data.


## ⚠️ Caution
- Be careful while editing the prompt instructions of the agents, not being specific can sometimes lead to malfunction of handoffs.