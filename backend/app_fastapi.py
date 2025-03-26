from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import asyncio
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFacePipeline
from pydantic import BaseModel
from langchain_core.messages import HumanMessage

# Import the compiled agent from myagent.py
from code.src.myagent import agent


app = FastAPI()

class Transaction(BaseModel):
    transaction_id: str
    date: str
    account_number: str
    bank_name: str
    bank_statement_amount: str
    book_records_amount: str
    match_status: str
    break_reason: str


@app.post("/invoke_agent")
async def invoke_agent(transaction: Transaction):
    try:
        human_message_template = f"""\
### Transaction Details  
- **ID**: {transaction.transaction_id}  
- **Account**: {transaction.account_number}  
- **Bank**: {transaction.bank_name}  
- **Bank Statement Amount**: {transaction.bank_statement_amount}  
- **Book Records Amount**: {transaction.book_records_amount}
- **Date**: {transaction.date} 

### Required Actions And Output Format
1. Tool chain execution along with explanation 
2. Mention each tool call request and response (Show tool call in xml format)
3. Reconciliation status with confidence score 
4. Plain English summary for auditors  
"""
        messages = [HumanMessage(content=human_message_template)]
        # Run the compiled agent synchronously in an executor if needed
        result = await asyncio.get_running_loop().run_in_executor(None, lambda: agent.invoke({"messages": messages}))
        # Extract content from each message returned by the agent
        response = [msg.content for msg in result["messages"]]
        return {"result": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
