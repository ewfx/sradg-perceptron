import faiss
import numpy as np
import pandas as pd
from preprocess import load_and_preprocess_data

def build_vector_index(data: pd.DataFrame, feature_cols: list):
    """
    Build FAISS vector index from feature columns.
    
    Args:
        data (pd.DataFrame): Input DataFrame
        feature_cols (list): List of column names to use as features
        
    Returns:
        faiss.Index: Built FAISS index
    """
    vectors = data[feature_cols].values.astype('float32')
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)
    return index

def query_similar_cases(index: faiss.Index, query_vector: np.ndarray, historical_data: pd.DataFrame, k: int = 5):
    """
    Query similar historical cases using vector search.
    
    Args:
        index (faiss.Index): Built FAISS index
        query_vector (np.ndarray): Vector to query
        historical_data (pd.DataFrame): Historical data
        k (int): Number of similar cases to return
        
    Returns:
        pd.DataFrame: Similar historical cases
    """
    query_vector = query_vector.astype('float32').reshape(1, -1)
    distances, indices = index.search(query_vector, k)
    return historical_data.iloc[indices[0]]

def main():
    # Load and preprocess data
    historical_data, current_data = load_and_preprocess_data(
        'data/synthetic_historical_report.csv',
        'data/synthetic_current_report.csv'
    )
    
    # Define feature columns for vector search
    feature_cols = ['Account Number', 'Bank Statement Amount', 'Book Records Amount', 'Amount_Difference']
    
    # Build search index
    index = build_vector_index(historical_data, feature_cols)
    
    # Example query
    current_vector = current_data.iloc[1][feature_cols].values
    similar_cases = query_similar_cases(index, current_vector, historical_data)
    
    print("Current Vector")
    print(current_data.iloc[1])
    print("\nSimilar Cases")
    print(similar_cases)

if __name__ == "__main__":
    main()
