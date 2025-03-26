import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

def load_and_preprocess_data(historical_file: str, current_file: str):
    """
    Load and preprocess historical and current transaction data.
    
    Args:
        historical_file (str): Path to historical data CSV
        current_file (str): Path to current data CSV
        
    Returns:
        tuple: Preprocessed historical and current data DataFrames
    """
    historical_data = pd.read_csv(historical_file)
    current_data = pd.read_csv(current_file)

    # Calculate amount differences
    historical_data["Break Reason"] = historical_data["Break Reason"].fillna("No Break")
    historical_data['Amount_Difference'] = historical_data['Bank Statement Amount'] - historical_data['Book Records Amount']
    current_data['Amount_Difference'] = current_data['Bank Statement Amount'] - current_data['Book Records Amount']

    # Encode account numbers
    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=951048)
    historical_data['Account Number'] = encoder.fit_transform(historical_data[['Account Number']])
    current_data['Account Number'] = encoder.transform(current_data[['Account Number']])

    return historical_data, current_data