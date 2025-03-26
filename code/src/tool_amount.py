from tools import ReconciliationTools
from preprocess import load_and_preprocess_data

historical_file = "data/synthetic_historical_report.csv"
current_file = "data/synthetic_current_report.csv"
# Sample historical CSV data
historical_data, current_data = load_and_preprocess_data(historical_file, current_file)

tool = ReconciliationTools(historical_data, current_data)
ans = tool.get_prediction(current_data.iloc[0].to_dict())
print(ans)