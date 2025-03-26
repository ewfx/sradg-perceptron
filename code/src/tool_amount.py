from tools import ReconciliationTools
from statistical import StatisticalAnalysisTool
from preprocess import load_and_preprocess_data

historical_file = "data/synthetic_historical_report.csv"
current_file = "data/synthetic_current_report.csv"
# Sample historical CSV data
historical_data, current_data = load_and_preprocess_data(historical_file, current_file)

tool = StatisticalAnalysisTool(historical_data, current_data)
ans = tool.statistical_analysis(transaction_id='TXN-000005')
print(ans)