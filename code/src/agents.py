from langchain.chat_models import ChatOpenAI
from langchain.tools import Tool
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent, AgentType
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import requests

load_dotenv()
# Load LLM
import google.generativeai as genai

# Set up Gemini API
genai.configure(api_key="AIzaSyA56lmNNt9Bwb0QYDfyiu_eK169Me2uNVE")  # Replace with your Gemini API key

def load_llm():
    return ChatGroq(model_name="deepseek-r1-distill-llama-70b")

# Initialize the Gemini model
#model = genai.GenerativeModel('gemini-2.0-flash-lite')
# llm = genai.GenerativeModel('gemini-2.0-flash-lite')#ChatOpenAI(model_name="gpt-4-turbo")


# llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite")
llm = load_llm()

# Memory for storing past anomalies & actions
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Define API-based tools for reconciliation
def query_bank(transaction_id):
    """Query the bank for transaction details."""
    response = requests.get(f"https://bankapi.com/transactions/{transaction_id}")
    return response.json()

def update_ledger(transaction_id, amount, reason):
    """Correct ledger entries automatically."""
    payload = {"transaction_id": transaction_id, "amount": amount, "reason": reason}
    response = requests.post("https://accountingapi.com/update_entry", json=payload)
    return response.json()

def trigger_rpa_bot(action, transaction_id):
    """Trigger RPA bot for UI-based reconciliation actions."""
    response = requests.post("https://rpa-bot.com/execute", json={"action": action, "transaction_id": transaction_id})
    return response.json()

def escalate_to_finance(transaction_id, reason):
    """Escalate to the finance team for manual review."""
    return f"Finance team notified for transaction {transaction_id}: {reason}"

# Register tools for LangChain Agent
tools = [
    Tool(name="Query Bank", func=query_bank, description="Fetch transaction details from the bank."),
    Tool(name="Update Ledger", func=update_ledger, description="Fix discrepancies in company ledger."),
    Tool(name="Trigger RPA", func=trigger_rpa_bot, description="Use RPA bot for manual reconciliation."),
    Tool(name="Escalate to Finance", func=escalate_to_finance, description="Notify finance team for review.")
]

agent = initialize_agent(
    agent=AgentType.OPENAI_FUNCTIONS,
    tools=tools,
    llm=llm,
    memory=memory,
    verbose=True
)

def handle_anomaly(anomaly_type, justification, transaction_id):
    """AI-driven reconciliation decision-making and execution."""

    prompt = f"""
    You are an intelligent reconciliation agent. A financial anomaly has been detected:
    - **Anomaly Type**: {anomaly_type}
    - **Justification**: {justification}
    - **Transaction ID**: {transaction_id}

    Based on the justification, decide the best action:
    1. Query Bank - If more details from the bank statement are needed.
    2. Update Ledger - If the company’s records need to be corrected.
    3. Trigger RPA - If a manual reconciliation process is required.
    4. Escalate to Finance - If the issue requires human intervention (e.g., fraud).

    **Use the available tools to take action.**
    """

    decision = agent.invoke(prompt)
    return decision  # The action is automatically executed by the agent.

# Example Call
response = handle_anomaly(
    anomaly_type="Missing Transaction",
    justification="Transaction exists in company books but is missing in the bank statement.",
    transaction_id="TXN12345"
)

print(response)
