import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import uuid
from langchain_core.tools import tool
import random

class DetailTool:
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