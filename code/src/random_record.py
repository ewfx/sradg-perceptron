from preprocess import load_and_preprocess_data

## Load the Data
historical_file = "data/synthetic_historical_report.csv"
current_file = "data/synthetic_current_report.csv"
historical_data, current_data = load_and_preprocess_data(historical_file, current_file)

# write a function to return a record randomly from the current data in form a dictionary
def record():
    return current_data.sample(1).to_dict(orient='records')[0]