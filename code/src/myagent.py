from langchain_core.tools import tool
from dotenv import load_dotenv
from typing_extensions import Literal
from langgraph.graph import MessagesState
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from typing_extensions import TypedDict
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from preprocess import load_and_preprocess_data
from tools_ import ReconciliationTools
from tools.query import HistoricalDataQueryTool
from tools.statistical import StatisticalAnalysisTool
from tools.add import AddTransactionTool
from tools.alert import AlertTool
from tools.update import UpdateRecordTool
from tools.cur_converter import CurrencyConvertorTool
from tools.details import DetailTool
from tools.escalate import EscalateTool
from tools.prediction import PredictionTool
from tools.remove import RemoveTransactionTool
from random_record import record

# Load the transaction
trans = record()

## Load the LLM
load_dotenv()

llm = ChatGroq(model="deepseek-r1-distill-qwen-32b", model_kwargs={
        "max_completion_tokens": 4000
    })

## Load the Data
historical_file = "data/synthetic_historical_report.csv"
current_file = "data/synthetic_current_report.csv"
historical_data, current_data = load_and_preprocess_data(historical_file, current_file)

# Initialize tools
recon_tools = ReconciliationTools(historical_data, current_data)
query_tool = HistoricalDataQueryTool(historical_data, current_data)
stat_tool = StatisticalAnalysisTool(historical_data, current_data)
add_tool = AddTransactionTool(historical_data, current_data)
alert_tool = AlertTool(historical_data, current_data)
update_tool = UpdateRecordTool(historical_data, current_data)
cur_converter = CurrencyConvertorTool(historical_data, current_data)
detail_tool = DetailTool(historical_data, current_data)
escalate_tool = EscalateTool(historical_data, current_data)
get_prediction = PredictionTool(historical_data, current_data)
remove_tool = RemoveTransactionTool(historical_data, current_data)

# Augment the LLM with tools
tools = [query_tool.query,
         stat_tool.statistical_analysis,
         add_tool.add_transaction,
         alert_tool.human_alert,
         update_tool.update_amount,
         cur_converter.currency_conversion,
         detail_tool.get_more_details,
         escalate_tool.escalate_to_finance,
         get_prediction.get_prediction,
         remove_tool.remove_transaction]

tools_by_name = {tool.name: tool for tool in tools}
llm_with_tools = llm.bind_tools(tools)

trans['Account Number'] = 'ACCT-611080'
time_diff_threshold = 3  # From your TimeDifferenceAgent
amt_variance_threshold = 100  
percent_variance_threshold = 2

def llm_call(state: MessagesState):
    """LLM decides whether to call a tool or not"""
    
    system_prompt = f"""\
    # Role: Reconciliation AI Agent
    **Specialization**: Transaction auditing & anomaly resolution
    **Primary Objective**: Automate reconciliation while minimizing false positives

    ## Core Capabilities
    1. **Anomaly Detection**:
    - Timing mismatches
    - Extra or Missing Transactions
    - Amount variances
    - Fraud pattern recognition

    2. **Action Framework**:
    - Auto-resolution for confidence ≥90%
    - Escalation required for fraud probability >40%
    - Documentation requests on missing/incomplete data

    ## Compliance Protocol
    - **Audit**: Full tool invocation logging

    ## Failure Modes
    - If uncertain: Escalate via `get_more_details()` or 
    - call `human_alert()` when getting undesireable results from tools and need manual intervention and ABORT"""

    return {
        "messages": [
            llm_with_tools.invoke(
                [
                    SystemMessage(content=system_prompt)
                ] + state["messages"]
            )
        ]
    }


def tool_node(state: dict):
    """Performs the tool call"""

    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
    return {"messages": result}


# Conditional edge function to route to the tool node or end based upon whether the LLM made a tool call
def should_continue(state: MessagesState) -> Literal["environment", END]:
    """Decide if we should continue the loop or stop based upon whether the LLM made a tool call"""

    messages = state["messages"]
    last_message = messages[-1]
    # If the LLM makes a tool call, then perform an action
    if last_message.tool_calls:
        return "Action"
    # Otherwise, we stop (reply to the user)
    return END


# Build workflow
agent_builder = StateGraph(MessagesState)

# Add nodes
agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("environment", tool_node)

# Add edges to connect nodes
agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges(
    "llm_call",
    should_continue,
    {
        # Name returned by should_continue : Name of next node to visit
        "Action": "environment",
        END: END,
    },
)
agent_builder.add_edge("environment", "llm_call")

# Compile the agent
agent = agent_builder.compile()

# trans = {
#     'Transaction ID': '3156.0',
#     'Date': '2040-02-13',
#     'Account Number': 'ACCT-611080',
#     'Bank Name': 'Bank of America',
#     'Bank Statement Amount': '-4.0030627',
#     'Book Records Amount': '-4.087139',
#     'Match Status': 'Break',
#     'Amount_Difference': -329.43
# }
# print(trans)

human_message_template = f"""\
### Transaction Details  
- **ID**: {trans['Transaction ID']}  
- **Account**: {trans['Account Number']}
- **Bank Name**: {trans['Bank Name']}  
- **Bank Statement Amount**: {trans['Bank Statement Amount']}  
- **Book Records Amount**: {trans['Book Records Amount']}
- **Amount Difference**: {trans['Amount_Difference']}
- **Date**: {trans['Date']} 

### Required Actions And Output Format
1. Tool chain execution along with explanation 
2. Mention each tool call request and response (Show tool call in xml format)
3. Reconciliation status with confidence score 
4. Plain English summary for auditors  
""" 

# print(human_message_template)

messages = [HumanMessage(content=human_message_template)]

messages = agent.invoke({"messages": messages})
for m in messages["messages"]:
    m.pretty_print()