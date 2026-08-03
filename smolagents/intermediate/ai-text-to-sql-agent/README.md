# 🧑‍💻 AI Text-to-SQL Agent (Smol Agents)

A user-friendly agent that converts natural language questions into SQL queries and executes them on your data. Built with [Hugging Face Smol Agents](https://huggingface.co/docs/smolagents/en/index) and [Streamlit](https://streamlit.io/), this app lets you upload your own CSV data or try ready-to-use demo datasets. The agent leverages LLMs to reason about your schema and generate accurate SQL queries, with results and queries shown in the UI.

## ✨ Features

- **Text-to-SQL**: Converts natural language to SQL and executes on your data
- **Upload Your Data**: Supports CSV uploads for custom queries
- **Demo Datasets**: Try with built-in example tables
- **LLM Reasoning**: Uses Smol Agents and OpenAI API
- **Streamlit UI**: Simple, interactive web interface

## 🛠️ Prerequisites

- Python 3.8+
- Sign up for an OpenAI account and generate an API key at [OpenAI API Keys](https://platform.openai.com/account/api-keys).
- Jupyter or Google Colab for running the notebook

## 📥 Installation

1. Clone the repository and navigate to this directory:
   ```bash
   git clone https://github.com/thejenilsoni/master-ai-agents.git
   cd master-ai-agents/smolagents/intermediate/ai-text-to-sql-agent
   ```

## 🌐 Run via Google Colab or Jupyter + Streamlit (with localtunnel)

You can run this agent in Google Colab or Jupyter and access the Streamlit UI remotely using [localtunnel](https://theboroer.github.io/localtunnel-www/). 

### Steps:
1. **Open the python notebook and run the setup cells** to install dependencies and write the Streamlit app file.
2. **Start Streamlit and localtunnel** by running the provided cell:
   ```bash
   !streamlit run /content/ai_text_to_sql_agent.py &>/content/logs.txt & npx localtunnel --port 8501 & curl ipv4.icanhazip.com
   ```
3. **Read two things out of that cell's output.** It prints both, one after
   the other:

   - a localtunnel URL of the form `https://<random-words>.loca.lt`
   - a bare IPv4 address on its own line, e.g. `34.16.203.11` — this is the
     output of `curl ipv4.icanhazip.com`

4. **Open the URL, and paste the IP address as the password.** localtunnel
   shows a "Tunnel Password" interstitial before it will forward you to the app.
   The password is that IP address and nothing else — no `https://`, no port.

   This is the step people get stuck on: the page asks for a password, the
   notebook never uses the word "password", and the value it wants is sitting
   further up the output looking like an unrelated diagnostic.

5. **Use the Streamlit interface.** Upload a CSV or pick a demo dataset, paste
   your OpenAI API key into the sidebar, and ask a question in plain English —
   "which five customers spent the most last quarter?". The app shows the SQL it
   generated above the result table, so you can check the query before trusting
   the answer.


## 🧠 Thought Process of the agent
- A file named `logs.txt` will be generated(in the Google Colab / Jupyter directory that is in use), which contains the complete thought process of the agent, starting from the query to the end result.

## 🧩 Frameworks & Tools Used
- [Smol Agents](https://huggingface.co/docs/smolagents/en/index)
- [Streamlit](https://streamlit.io/)
- [SQLAlchemy](https://www.sqlalchemy.org/)
- [pandas](https://pandas.pydata.org/)

## 🛠️ Troubleshooting
- **Token/Permission Errors**: Ensure your OpenAI API key is valid.

## 📚 Learn More
- [Smol Agents Documentation](https://huggingface.co/docs/smolagents/en/examples/text_to_sql)
- [Streamlit Documentation](https://docs.streamlit.io/)

---

Feel free to explore, use, and extend this agent for your own data and use cases! 