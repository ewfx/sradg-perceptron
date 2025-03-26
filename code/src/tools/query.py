import pandas as pd
import sqlite3
from io import StringIO
from typing import Union, List, Dict
from langchain_core.tools import tool

class HistoricalDataQueryTool:
    def __init__(self, mydf: pd.DataFrame, current_data: pd.DataFrame):
        """
        Initialize with CSV data and create in-memory SQL database
        :param csv_data: Raw CSV string of historical data
        """
        self.df = mydf.copy()
        self._sanitize_columns()
        self.conn = sqlite3.connect(":memory:")
        self.df.to_sql('historical_data', self.conn, index=False)

    def _sanitize_columns(self):
        """Clean column names for SQL compatibility"""
        self.df.columns = [col.replace(' ', '_').replace('-', '_').lower() 
                          for col in self.df.columns]

    @tool
    def query(self, sql: str) -> Union[List[Dict], str]:
        """
        Execute safe SQL query on historical data to retrieve records
        
        Parameters:
        - query: SELECT * from historical_data where {filters}
        
        Historical Database Column Names: ['transaction_id', 'date', 'account_number', 'bank_name', 'bank_statement_amount', 'book_records_amount', 'match_status', 'break_reason', 'amount_difference']

        Returns:
        List of records or error message
        
        Example:
        >>> query(SELECT * from historical_data where "account_number='3156.0' and amount_difference > 100")
        """
        try:
            clean_sql = sql.replace(';', '').strip()
            # Enforce limits
            if "limit" not in clean_sql.lower():
                clean_sql += " LIMIT 5"

            # Execute query
            result = pd.read_sql(clean_sql, self.conn)
            
            # Convert to list of dicts
            return result.to_dict(orient='records')

        except Exception as e:
            return f"Query failed: {str(e)}"