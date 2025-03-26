from langchain_groq import ChatGroq
from dotenv import load_dotenv
from preprocess import load_and_preprocess_data
from vector_search import build_vector_index
from vector_search import query_similar_cases
from categorizer import predict_break_reason, predict_single_record
import cloudpickle
import joblib


load_dotenv()

def load_llm():
    return ChatGroq(model_name="llama-3.3-70b-versatile")

def load_model(filename='encoder.pkl'):
    with open(filename, 'rb') as f:
        model = cloudpickle.load(f)
    return model

def generate_comments(similar_cases):
    # Prepare the prompt with clear instructions for specific outputs
    prompt = f"""
    Transaction Details:
    - Transaction ID: {transaction['Transaction ID']}
    - Date: {transaction['Date']}
    - Account Number: {transaction['Account Number']}
    - Bank Name: {transaction['Bank Name']}
    - Bank Statement Amount: {transaction['Bank Statement Amount']}
    - Book Records Amount: {transaction['Book Records Amount']}
    - Amount Difference: {transaction['Amount_Difference']}

    Similar Historical Cases:
    {similar_cases.to_string()}

    Task:
    Analyze the transaction and identify the **exact reason** for the anomaly. Then, suggest **specific corrective actions** to resolve it.
    Follow this format strictly:
    1. **Anomaly Classification**: (e.g., Amount Mismatch, Timing Difference, Fraud, Duplicate Entry, Missing Transaction, etc.)
    2. **Reason**: (Explain the exact cause of the anomaly based on the transaction details and historical cases.)
    3. **Corrective Action**: (Provide a specific, actionable step to resolve the anomaly.)
    """

    # Generate response using Gemini
    response = llm.generate_content(prompt)
    return response.text


llm = load_llm()
historical_data, current_data = load_and_preprocess_data(
    'data/synthetic_historical_report.csv',
    'data/synthetic_current_report.csv'
)
output_file = "predicted_break_reason.csv"
feature_cols = ['Account Number', 'Bank Statement Amount', 'Book Records Amount', 'Amount_Difference']
index = build_vector_index(historical_data, feature_cols)
current_vector = current_data.iloc[0][feature_cols].values
similar_cases = query_similar_cases(index, current_vector, historical_data)
print(similar_cases)

model = load_model('models/ml-models/xgb_pipeline_model.pkl')
encoder = load_model('models/ml-models/label_encoder.pkl')

# model = joblib.load('models/ml-models/xgb_pipeline_model.pkl')
# encoder = joblib.load('models/ml-models/label_encoder.pkl')
feature_cols = ["Date", "Account Number", "Bank Name",
            "Bank Statement Amount", "Book Records Amount", "Match Status"]
    
anomaly = predict_single_record(model, encoder, feature_cols, current_data.iloc[0].to_dict())

print(anomaly)