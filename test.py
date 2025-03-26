from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline


# model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")
# tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")

# model.save_pretrained("models/generative/flan-t5-large")
# tokenizer.save_pretrained("models/tokenizers/flan-t5-large")

# model = AutoModelForSeq2SeqLM.from_pretrained("models/generative/flan-t5-large")
# tokenizer = AutoTokenizer.from_pretrained("models/tokenizers/flan-t5-large")

pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer)


llm = HuggingFacePipeline(pipeline=pipe)
# Example usage
prompt_template = "Answer the following question: {question}"
chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate.from_template(prompt_template)
)

# Run the model
result = chain.run({"question": "What is machine learning?"})
print(result)
