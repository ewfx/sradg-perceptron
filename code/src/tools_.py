import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import uuid
from langchain_core.tools import tool
from sklearn.ensemble import IsolationForest  # Example model
from helper.categorizer import predict_single_record
import random
import cloudpickle

class ReconciliationTools:
    def __init__(self, historical_data: pd.DataFrame, current_data: pd.DataFrame):
        self.historical_data = historical_data
        self.current_data = current_data
        self._prepare_data()
        
        # Initialize mock ML model
        # self.model = self._train_mock_model()

    def _prepare_data(self):
        """Convert CSV data to SQL-like structure"""
        self.historical_data['Date'] = pd.to_datetime(self.historical_data['Date'])
        self.current_data['Date'] = pd.to_datetime(self.current_data['Date'])
        self.merged_data = pd.concat([self.historical_data, self.current_data])

    def _train_mock_model(self):
        """Train a simple anomaly detection model"""
        features = ['Bank Statement Amount', 'Book Records Amount']
        model = IsolationForest(contamination=0.01)
        model.fit(self.historical_data[features].fillna(0))
        return model

    @tool
    def human_alert(self, transaction_id: str) -> str:
        """
        Send a human alert message when any other tool is not responding properly or unsure of further action.
        """
        return f"ABORTING: Human intervention required to resolve issue with transaction {transaction_id}"

    @tool
    def update_amount(self, transaction_id: str, date: str, record: str, amount: float, reason: str) -> Dict[str, Any]:
        """
        Update the bank statement amount for a transaction.
        
        Parameters:
        - transaction_id: ID of transaction to adjust
        - date: Date of transaction
        - record: 'bank' or 'book' to specify which record to update
        - amount: New corrected amount
        - reason: Reason for adjustment
        
        Returns:
        Confirmation with adjustment details
        
        Example:
        >>> update_ledger('TXN123', '2040-02-13', 1500.0, 'Fix currency conversion error')
        """
        return {
            'status': 'success',
            'record': record,
            'message': f"Ledger updated for {transaction_id} on {date}",
            'adjustment_id': f"ADJ-{uuid.uuid4().hex[:6]}",
            'new_amount': amount,
            'reason': reason
        }
    
    
    @tool
    def remove_transaction(self, transaction_id: str, date: str, record: str, reason: str) -> Dict[str, Any]:
        """
        Remove a transaction from the current data.
        
        Parameters:
        - transaction_id: ID of transaction to remove
        - date: Date of transaction
        - record: 'bank' or 'book' to specify which record to update
        - reason: Reason for removal
        
        Returns:
        Confirmation with removal details
        
        Example:
        >>> remove_transaction('TXN456', 'Duplicate transaction')
        """
        return {
            'status': 'success',
            'record': record,
            'message': f"Transaction {transaction_id} removed on {date}",
            'reason': reason
        }
    
    @tool
    def add_transaction(self, transaction_id: str, date: str, record: str, amount: float, reason: str) -> Dict[str, Any]:
        """
        Add a new transaction to the current data.
        
        Parameters:
        - transaction_id: ID of new transaction
        - date: Date of transaction
        - record: 'bank' or 'book' to specify which record to update
        - amount: New transaction amount
        - reason: Reason for addition
        
        Returns:
        Confirmation with addition details
        
        Example:
        >>> add_transaction('TXN789', '2040-02-13', 1500.0, 'New transaction')
        """
        return {
            'status': 'success',
            'record': record,
            'message': f"Transaction {transaction_id} added on {date}",
            'new_amount': amount,
            'reason': reason
        }

    @tool
    def escalate_to_finance(self, transaction_id: str, date: str, reason: str) -> Dict[str, Any]:
        """
        Finance team escalation by generating a service ticket.
        
        Parameters:
        - transaction_id: ID of problematic transaction
        - date: Date of transaction
        - reason: Description of the issue
        
        Returns:
        Mock service ticket details
        
        Example:
        >>> escalate_to_finance('TXN456', 'Suspected duplicate transaction')
        """
        return {
            'ticket_id': f"FIN-{uuid.uuid4().hex[:6]}",
            'transaction_id': transaction_id,
            'date': date,
            'status': 'open',
            'priority': 'high' if 'fraud' in reason.lower() else 'medium',
            'assigned_to': 'finance-team@company.com',
            'message': f"Escalation created: {reason}"
        }

    @tool
    def currency_conversion(self, transaction_id: str, amount: float, new_rate: float) -> Dict[str, Any]:
        """
        Currency conversion adjustments.
        
        Parameters:
        - transaction_id: ID of foreign currency transaction
        - amount: Original amount in foreign currency
        - new_rate: Exchange rate to apply
        
        Returns:
        Conversion audit record
        
        Example:
        >>> currency_conversion('TXN789', 1.12)
        """
        
        return {
            'transaction_id': transaction_id,
            'original_amount': amount,
            'new_rate': new_rate,
            'converted_amount': amount * new_rate,
            'audit_trail': f"Rate updated from {amount/new_rate:.2f} to {new_rate}"
        }
    

    @tool
    def get_more_details(self, transaction_id: str) -> Dict[str, Any]:
        """
        Request finance team to check the receipt and confirm if it is fradulent or it is a mismatch in amount.
        
        Parameters:
        - transaction_id: ID of transaction needing clarification
        
        Returns:
        Mock request confirmation
        
        Example:
        >>> get_more_details('TXN101')
        """
        return {
            'request_id': f"INFO-{uuid.uuid4().hex[:6]}",
            'transaction_id': transaction_id,
            'requested_documents': ['receipt', 'counterparty_history'],
            'status': 'Success',
            'message': f"Details requested from finance team for {transaction_id}",
            'response': random.choice(['Fradulent', 
                                       'Amount Mismatch: Coorect Bank Amount',
                                       'Amount Mismatch: Correct Book Amount'])
        }

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