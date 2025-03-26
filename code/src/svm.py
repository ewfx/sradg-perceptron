import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM

def fit_models(historical_data, contamination=0.1, nu=0.1):
    """
    Fit statistical and ML models for anomaly detection on historical data.
    
    Args:
        historical_data (pd.DataFrame): Historical transaction data
        contamination (float): Contamination parameter for Isolation Forest
        nu (float): Nu parameter for One-Class SVM
        
    Returns:
        dict: Dictionary containing fitted models and parameters
    """
    historical_amount_diff = historical_data['Amount_Difference'].values
    
    # Calculate IQR parameters
    Q1 = np.percentile(historical_amount_diff, 25)
    Q3 = np.percentile(historical_amount_diff, 75)
    IQR = Q3 - Q1

    # Fit Isolation Forest
    iso_forest_model = IsolationForest(contamination=contamination)
    iso_forest_model.fit(historical_amount_diff.reshape(-1, 1))

    # Fit One-Class SVM
    one_class_svm_model = OneClassSVM(nu=nu)
    one_class_svm_model.fit(historical_amount_diff.reshape(-1, 1))
    
    return {
        'Q1': Q1,
        'Q3': Q3,
        'IQR': IQR,
        'isolation_forest': iso_forest_model,
        'one_class_svm': one_class_svm_model
    }

def predict_isolation_forest(model, data):
    """
    Predict anomalies using Isolation Forest model.
    
    Args:
        model: Fitted Isolation Forest model
        data (array-like): Data to predict on
        
    Returns:
        array: Predictions (-1 for anomalies, 1 for normal)
    """
    return model.predict(data.reshape(-1, 1))

def predict_one_class_svm(model, data):
    """
    Predict anomalies using One-Class SVM model.
    
    Args:
        model: Fitted One-Class SVM model
        data (array-like): Data to predict on
        
    Returns:
        array: Predictions (-1 for anomalies, 1 for normal)
    """
    return model.predict(data.reshape(-1, 1))

def predict_iqr(Q1, Q3, IQR, data, k=1.5):
    """
    Detect anomalies using IQR method.
    
    Args:
        Q1 (float): First quartile
        Q3 (float): Third quartile
        IQR (float): Interquartile range
        data (array-like): Data to predict on
        k (float): IQR multiplier for outlier detection
        
    Returns:
        array: Predictions (-1 for anomalies, 1 for normal)
    """
    lower_bound = Q1 - k * IQR
    upper_bound = Q3 + k * IQR
    
    predictions = np.ones(len(data))
    predictions[(data < lower_bound) | (data > upper_bound)] = -1
    return predictions

def main():
    
    # Load historical and current reconciliation data
    historical_data = pd.read_csv('data/synthetic_historical_report.csv')
    current_data = pd.read_csv('data/synthetic_current_report.csv')

    # Standardize data formats
    historical_data['Date'] = pd.to_datetime(historical_data['Date'])
    current_data['Date'] = pd.to_datetime(current_data['Date'])

    # Handle missing values
    historical_data.fillna({'Break Reason': 'Unknown', 'Resolution Status': 'Unresolved'}, inplace=True)
    current_data.fillna({'Match Status': 'Unknown'}, inplace=True)

    # Extract key features
    historical_data['Amount_Difference'] = historical_data['Bank Statement Amount'] - historical_data['Book Records Amount']
    current_data['Amount_Difference'] = current_data['Bank Statement Amount'] - current_data['Book Records Amount']
    
    # Fit models
    models = fit_models(historical_data)
    
    # Get one record from current data for testing
    test_record = current_data.iloc[0]
    test_amount_diff = test_record['Amount_Difference']
    
    print("\nTesting predictions on record:")
    print(f"Transaction ID: {test_record['Transaction ID']}")
    print(f"Amount Difference: {test_amount_diff:.2f}")
    
    # Predict with each model
    print("\nPredictions:")
    
    # One-Class SVM prediction
    svm_pred = predict_one_class_svm(models['one_class_svm'], np.array([test_amount_diff]))
    print(f"One-Class SVM: {'Anomaly' if svm_pred[0] == -1 else 'Normal'}")

    #predict with isolation forest
    iso_pred = predict_isolation_forest(models['isolation_forest'], np.array([test_amount_diff]))
    print(f"Isolation Forest: {'Anomaly' if iso_pred[0] == -1 else 'Normal'}")
    
    # IQR prediction
    iqr_pred = predict_iqr(
        models['Q1'], 
        models['Q3'],
        models['IQR'],
        np.array([test_amount_diff])
    )
    print(f"IQR Method: {'Anomaly' if iqr_pred[0] == -1 else 'Normal'}")

if __name__ == "__main__":
    main()