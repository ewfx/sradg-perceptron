from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import asyncio
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFacePipeline

app = FastAPI()

model = None
tokenizer = None
pipe = None
llm = None
chain = None

@app.on_event("startup")
async def load_model():
    global model, tokenizer, pipe, llm, chain
    # Optionally, run the blocking call in a thread pool:
    loop = asyncio.get_running_loop()
    model, tokenizer = await loop.run_in_executor(None, lambda: (
        AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large"),
        AutoTokenizer.from_pretrained("google/flan-t5-large")
    ))
    
    pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer)
    llm = HuggingFacePipeline(pipeline=pipe)
    
    prompt_template = """
    Process the following CSV data and generate the output CSV.
    Historical CSV:
    {historical}
    
    Current CSV:
    {current}
    
    After the CSV output, on a new line output "ACTIONS:" followed by a summary of actions.
    """
    chain = LLMChain(
        llm=llm,
        prompt=PromptTemplate.from_template(prompt_template)
    )
    print("Model loaded successfully.")

output_csv_content = None
actions_taken = None

@app.post("/process")
async def process_files(
    historical_file: UploadFile = File(...),
    current_file: UploadFile = File(...)
):
    global output_csv_content, actions_taken, chain
    if chain is None:
        raise HTTPException(status_code=503, detail="Model is still loading.")
    try:
        hist_bytes = await historical_file.read()
        curr_bytes = await current_file.read()
        historical_csv = hist_bytes.decode("utf-8")
        current_csv = curr_bytes.decode("utf-8")
        
        prompt_vars = {"historical": historical_csv, "current": current_csv}
        result = chain.run(prompt_vars)
        
        if "ACTIONS:" in result:
            output_part, actions_part = result.split("ACTIONS:", 1)
        else:
            output_part = result
            actions_part = "No actions logged."
        
        output_csv_content = output_part.strip()
        actions_taken = actions_part.strip()
        
        with open("output_generated.csv", "w", encoding="utf-8") as f:
            f.write(output_csv_content)
        
        return JSONResponse({
            "message": "Processing complete.",
            "output": output_csv_content,
            "actions": actions_taken
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/result")
def get_result():
    if not output_csv_content:
        raise HTTPException(status_code=404, detail="No result found. Process files first.")
    return {"output_csv": output_csv_content, "actions": actions_taken}