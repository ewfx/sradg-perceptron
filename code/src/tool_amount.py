from tools.query import HistoricalDataQueryTool
from preprocess import load_and_preprocess_data

historical_file = "data/synthetic_historical_report.csv"
current_file = "data/synthetic_current_report.csv"
# Sample historical CSV data
historical_data, current_data = load_and_preprocess_data(historical_file, current_file)

data = {
    'Transaction ID': 'TXN-006253',
    'Date': '2040-02-13',
    'Account Number': 'ACCT-611080',
    'Bank Name': 'Bank of America',
    'Bank Statement Amount': '-4.0030627',
    'Book Records Amount': '-4.087139',
    'Match Status': 'Break',
    'Break Reason': 'Bank Error'
}

tool = HistoricalDataQueryTool(historical_data, current_data)
ans = tool.query("SELECT * from historical_data where account_number='3156.0'")
print(ans)