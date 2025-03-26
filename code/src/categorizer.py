import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from preprocess import load_and_preprocess_data
import joblib
import cloudpickle

# -------------------------------
# Custom Transformer for Date Features
# -------------------------------
class DateTransformer(BaseEstimator, TransformerMixin):
    """
    Transforms a date column into numerical features:
      - Year, Month, Day, Day of Week.
    """
    def __init__(self, date_format="%Y-%m-%d"):
        self.date_format = date_format

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Expecting X as a DataFrame or Series of date strings
        if isinstance(X, pd.DataFrame):
            dates = pd.to_datetime(X.iloc[:, 0], format=self.date_format, errors='coerce')
        else:
            dates = pd.to_datetime(X, format=self.date_format, errors='coerce')
        # Create DataFrame of date features
        df = pd.DataFrame({
            "Year": dates.dt.year,
            "Month": dates.dt.month,
            "Day": dates.dt.day,
            "DayOfWeek": dates.dt.dayofweek
        })
        return df.values

# -------------------------------
# Data Loading and Preprocessing Functions
# -------------------------------
def prepare_features_targets(df: pd.DataFrame, target_col: str, feature_cols: list):
    """Extract features and target from DataFrame. Label-encode the target."""
    X = df[feature_cols]
    y = df[target_col].astype(str)
    
    # Handle NaN in target just in case
    y = y.fillna("No break")
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    return X, y_encoded, le

# -------------------------------
# Pipeline Building Function
# -------------------------------
def build_pipeline():
    """Constructs the preprocessing and classification pipeline."""
    # Define column lists
    date_col = ["Date"]
    categorical_cols = ["Account Number", "Bank Name", "Match Status"]
    numeric_cols = ["Bank Statement Amount", "Book Records Amount"]

    # Date pipeline
    date_pipeline = Pipeline([
        ("date_transform", DateTransformer(date_format="%Y-%m-%d"))
    ])

    # Categorical pipeline: OneHotEncode
    categorical_pipeline = Pipeline([
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    # Numeric pipeline: StandardScaler
    numeric_pipeline = Pipeline([
        ("scaler", StandardScaler())
    ])

    # Combine into ColumnTransformer
    preprocessor = ColumnTransformer(transformers=[
        ("date", date_pipeline, date_col),
        ("cat", categorical_pipeline, categorical_cols),
        ("num", numeric_pipeline, numeric_cols)
    ])

    # Create full pipeline with XGBoost Classifier.
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", XGBClassifier(
            objective="multi:softmax",
            eval_metric="mlogloss",
            use_label_encoder=False,
            random_state=42
        ))
    ])

    return pipeline

# -------------------------------
# Model Training Function
# -------------------------------
def train_model(df_hist: pd.DataFrame, feature_cols: list, target_col: str):
    """Load data, train the pipeline, and return the trained model and LabelEncoder."""
    X, y, le = prepare_features_targets(df_hist, target_col, feature_cols)
    
    # Split the data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Build pipeline
    pipeline = build_pipeline()
    
    # Train the pipeline
    pipeline.fit(X_train, y_train)
    print("Training complete!")
    
    # Evaluate on validation set
    y_pred = pipeline.predict(X_val)
    print("Classification Report on Validation Set:")
    print(classification_report(y_val, y_pred, target_names=le.classes_))
    
    return pipeline, le

# -------------------------------
# Prediction Function
# -------------------------------
def predict_break_reason(model, label_encoder, df_current: pd.DataFrame, feature_cols: list, output_file: str):
    """Make predictions on current data and save the results."""
    X_current = df_current[feature_cols]
    
    # Predict with the trained pipeline (predictions are encoded)
    pred_encoded = model.predict(X_current)
    # Inverse transform to get original labels
    predictions = label_encoder.inverse_transform(pred_encoded)
    
    df_current["Predicted Break Reason"] = predictions
    df_current.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")

# -------------------------------
# Single Record Prediction Function
# -------------------------------
def predict_single_record(model, label_encoder, feature_cols, record_dict: dict) -> tuple:
    """
    Predict break reason for a single transaction record and return confidence score.
    
    Args:
        model: Trained model pipeline
        label_encoder: Fitted LabelEncoder
        record_dict: Dictionary containing transaction data with keys matching feature_cols:
                    "Date", "Account Number", "Bank Name", "Bank Statement Amount",
                    "Book Records Amount", "Match Status"
    
    Returns:
        tuple: (Predicted break reason, Confidence score)
    """
    # Convert single record to DataFrame
    record_df = pd.DataFrame([record_dict])
    record_df = record_df[feature_cols]
    # Make prediction
    pred_encoded = model.predict(record_df)
    prediction = label_encoder.inverse_transform(pred_encoded)[0]
    
    # Get confidence score (probability of the predicted class)
    pred_proba = model.predict_proba(record_df)
    confidence = pred_proba[0][pred_encoded[0]]
    
    return prediction, confidence

def save_model_cloudpickle(model, filename='model.pkl'):
    with open(filename, 'wb') as f:
        cloudpickle.dump(model, f)

# -------------------------------
# Main Function
# -------------------------------
def main():
    # File paths - adjust as necessary
    historical_file = "data/synthetic_historical_report.csv"
    current_file = "data/synthetic_current_report.csv"
    output_file = "predicted_break_reason.csv"
    
    # Define the columns used for training and target
    feature_cols = ["Date", "Account Number", "Bank Name",
                    "Bank Statement Amount", "Book Records Amount", "Match Status"]
    target_col = "Break Reason"

    historical_data, current_data = load_and_preprocess_data(historical_file, current_file)
    
    # Train model
    model, label_encoder = train_model(historical_data, feature_cols, target_col)
    
    save_model_cloudpickle(model, "models/ml-models/xgb_pipeline_model.pkl")
    save_model_cloudpickle(label_encoder, "models/ml-models/label_encoder.pkl")
    # Optionally, save the model and encoder for later use
    # joblib.dump(label_encoder, "models/ml-models/label_encoder.pkl")
    # joblib.dump(model, "models/ml-models/xgb_pipeline_model.pkl")
    
    # Predict on current data
    predict_break_reason(model, label_encoder, current_data, feature_cols, output_file)

if __name__ == "__main__":
    main()
