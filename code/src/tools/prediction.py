from typing import Dict, Any, Optional
import pandas as pd
from langchain_core.tools import tool
from helper.categorizer import predict_single_record
import cloudpickle

class PredictionTool:
    def __init__(self, historical_data: pd.DataFrame, current_data: pd.DataFrame):
        self.historical_data = historical_data
        self.current_data = current_data
        self._prepare_data()

    def _prepare_data(self):
        """Convert CSV data to SQL-like structure"""
        self.historical_data['Date'] = pd.to_datetime(self.historical_data['Date'])
        self.current_data['Date'] = pd.to_datetime(self.current_data['Date'])
        self.merged_data = pd.concat([self.historical_data, self.current_data])

    def load_model(self, filename='encoder.pkl'):
        with open(filename, 'rb') as f:
            model = cloudpickle.load(f)
        return model
    
    @tool
    def get_prediction(self, transaction_record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get Xg-Boost classification model prediction for anomaly type for the transaction.
        
        Parameters:
        - transaction_record: Dictionary of transaction features
        
        Returns:
        Anomaly score and risk assessment
        
        Example:
        >>> get_prediction({'Transaction ID': 'TXN-006253','Date': '2040-02-13','Account Number': 'ACCT-611080','Bank Name': 'Bank of America','Bank Statement Amount': '-4.0030627','Book Records Amount': '-4.087139','Match Status': 'Break'})
        """
        try:
            feature_cols = ["Date", "Account Number", "Bank Name",
            "Bank Statement Amount", "Book Records Amount", "Match Status"]
            model = self.load_model('models/ml-models/xgb_pipeline_model.pkl')
            encoder = self.load_model('models/ml-models/label_encoder.pkl')
            anomaly, confidence = predict_single_record(model, encoder, feature_cols, transaction_record)
            return {
                'transaction_id': transaction_record.get('Transaction ID', 'NEW'),
                'anomaly_score': anomaly,
                'confidence': confidence,
                'model_version': 'XGBoost_v1'
            }
        except Exception as e:
            return f"Prediction failed: {str(e)}"