import sqlite3
import pandas as pd

historical_file = "data/synthetic_historical_report.csv"
current_file = "data/synthetic_current_report.csv"
# Sample historical CSV data
historical_data = pd.read_csv(historical_file)

conn = sqlite3.connect(":memory:")
historical_data.to_sql('historical_data', conn, index=False)

clean_sql="SELECT * FROM historical_data WHERE Date='2040-02-13'"

clean_sql = clean_sql.strip().rstrip(';')
if "limit" not in clean_sql.lower():
    clean_sql += " LIMIT 5"

print(pd.read_sql(clean_sql, conn))s