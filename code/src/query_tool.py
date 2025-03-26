import pandas as pd
import sqlite3
from io import StringIO
from typing import Union, List, Dict
from langchain_core.tools import tool
from preprocess import load_and_preprocess_data

class HistoricalDataQueryTool:
    def __init__(self, mydf: pd.DataFrame):
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
        - filters: SELECT * from historical_data where {conditions passed in query}
        
        Returns:
        List of records or error message
        
        Example:
        >>> query("account_number='ACC123', amount > 100")
        """
        try:
            # Security checks
            if not sql.strip().lower().startswith("select"):
                return "Error: Only SELECT queries are allowed"

            # Enforce limits
            clean_sql = sql.strip().rstrip(';')
            if "limit" not in clean_sql.lower():
                clean_sql += " LIMIT 5"

            # Execute query
            print(self.df.head())
            result = pd.read_sql(clean_sql, self.conn)
            
            # Convert to list of dicts
            return result.to_dict(orient='records')

        except Exception as e:
            return f"Query failed: {str(e)}"

# Example Usage
if __name__ == "__main__":
    historical_file = "data/synthetic_historical_report.csv"
    current_file = "data/synthetic_current_report.csv"
    # Sample historical CSV data
    historical_data, current_data = load_and_preprocess_data(historical_file, current_file)
    # Initialize tool
    query_tool = HistoricalDataQueryTool(historical_data)

    # Test valid query
    print("Valid query results:")
    results = query_tool.query(
        "SELECT * FROM historical_data WHERE account_number='3156.0'"
    )
    print(results)

    # Test invalid query
    print("\nInvalid query results:")
    results = query_tool.query(
        "DELETE FROM historical_data WHERE account_number = 'ACC001'"
    )
    print(results)