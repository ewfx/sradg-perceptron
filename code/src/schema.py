from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
from typing import List
from dotenv import load_dotenv
import pandas as pd

load_dotenv()

def load_llm():
    return ChatGroq(model_name="llama-3.3-70b-versatile")


class Schema(BaseModel):
    key_columns: List[str] = Field(description="List of key columns that serve as unique identifiers")
    criteria_columns: List[str] = Field(description="List of numerical fields used for comparison, metrics, or calculations")
    date_columns: List[str] = Field(description="List of any columns containing dates, times, or temporal information")
    additional_columns: List[str] = Field(description="List of any other columns including comments and descriptions")


def detect_schema(data: pd.DataFrame) -> Schema:
    llm = load_llm()

    parser = PydanticOutputParser(pydantic_object=Schema)

    template = """
    You are an expert data analyst. Analyze this CSV structure and identify the purpose of each column:

    CSV SAMPLE:
    {sample_data}

    The column names in this dataset are: {column_names}

    Classify each column into exactly one of these types:
    - Key Columns: unique identifiers or primary keys
    - Criteria Columns: numerical fields used for comparison, metrics, or calculations
    - Date Columns: any columns containing dates, times, or temporal information
    - Additional Columns: all other columns including text descriptions, comments, statuses, etc.

    IMPORTANT: Every column must be classified into exactly one category. Don't leave any columns out.
    Respond using a JSON object with the following structure:
    {{
    "key_columns": ["col1", "col2", ...],
    "criteria_columns": ["col3", "col4", ...],
    "date_columns": ["col5", ...],
    "additional_columns": ["col6", "col7", ...]
    }}
    """
    
    prompt = PromptTemplate(
        template=template,
        input_variables=["sample_data", "column_names"],
    )

    structured_llm = llm.with_structured_output(Schema)

    response = structured_llm.invoke(prompt.format(sample_data=data.head().to_csv(), column_names=data.columns.tolist()))

    return {
        "key_columns": response.key_columns,
        "criteria_columns": response.criteria_columns,
        "date_columns": response.date_columns,
        "additional_columns": response.additional_columns
    }


if __name__ == "__main__":
    data = pd.read_csv("data/historical.csv")
    schema = detect_schema(data)
    print(schema)

