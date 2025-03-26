"""
Robust Adaptive Anomaly Detection Pipeline for Financial Reconciliation Reports
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from statsmodels.robust.scale import mad
from pycaret.anomaly import setup, create_model, assign_model
from sentence_transformers import SentenceTransformer
from umap import UMAP
import warnings

warnings.filterwarnings("ignore")

class AdaptiveAnomalyDetector:
    """
    Robust anomaly detection system that automatically adapts to:
    - Dataset size (small/large)
    - Data distribution characteristics
    - Schema variations in reconciliation reports
    """
    
    def __init__(self, schema, small_data_threshold=100, contamination=0.05):
        """
        Initialize with detected schema and configuration
        
        Args:
            schema (dict): Schema information from LLM analysis
            small_data_threshold (int): Minimum samples for ML approaches
            contamination (float): Expected anomaly proportion
        """
        self.schema = schema
        self.threshold = small_data_threshold
        self.contamination = contamination
        self.scaler = RobustScaler()
        self.text_encoder = SentenceTransformer('models/all-MiniLM-L6-v2')
        self._initialize_models()

    def _initialize_models(self):
        """Initialize detection models with dynamic parameters"""
        self.models = {
            'iforest': IsolationForest(
                contamination=self.contamination,
                random_state=42,
                bootstrap=True
            ),
            'lof': LocalOutlierFactor(
                novelty=True,
                contamination=self.contamination,
                n_neighbors='warn'
            ),
            'robust_covariance': EllipticEnvelope(
                contamination=self.contamination,
                random_state=42
            )
        }

    def _preprocess_data(self, df):
        """Adaptive preprocessing based on schema"""
        # Convert dates to numerical features
        date_features = []
        for col in self.schema.get('date_columns', []):
            df[f'{col}_doy'] = pd.to_datetime(df[col]).dt.dayofyear
            df[f'{col}_month'] = pd.to_datetime(df[col]).dt.month
            date_features.extend([f'{col}_doy', f'{col}_month'])
        
        # Select relevant columns
        numeric_cols = self.schema['criteria_columns'] + date_features
        text_cols = self.schema.get('comment_columns', [])
        print(numeric_cols)
        print(text_cols)
        # Create preprocessing pipeline
        preprocessor = ColumnTransformer([
            ('num', self.scaler, numeric_cols),
            ('text', TextEncoderTransformer(self.text_encoder), text_cols)
        ])
        
        # Fit and transform the data
        print(df[numeric_cols].shape)
        print(df[text_cols].shape)
        preprocessor.fit(df[numeric_cols + text_cols])
        return preprocessor.transform(df[numeric_cols + text_cols])

    def detect_anomalies(self, df):
        """
        Main detection method with automatic strategy selection
        
        Returns:
            tuple: (anomaly_scores, anomaly_labels, explanations)
        """
        X = self._preprocess_data(df)
        n_samples = X.shape[0]
        
        if n_samples < self.threshold:
            return self._handle_small_dataset(df, X)
        else:
            return self._handle_large_dataset(df, X)

    def _handle_small_dataset(self, df, X):
        """Statistical methods for small datasets"""
        # Modified Z-score method
        z_scores = self._calculate_modified_zscore(df)
        z_flags = np.any(np.abs(z_scores) > 3.5, axis=1)
        
        # Percentile-based detection
        percentile_flags = self._percentile_anomalies(df)
        
        # Ensemble results
        combined_score = 0.5*z_flags.astype(int) + 0.5*percentile_flags.astype(int)
        labels = combined_score > 0.5
        
        return combined_score, labels, self._generate_explanations(df, labels)

    def _handle_large_dataset(self, df, X):
        """Machine learning ensemble approach"""
        # PyCaret AutoML
        pycaret_preds = self._pycaret_detection(df)
        
        # Model ensemble
        ml_scores = self._ensemble_detection(X)
        
        # Text analysis
        text_scores = self._analyze_comments(df) if self.schema.get('comment_columns') else 0
        
        # Combine scores
        final_scores = 0.6*ml_scores + 0.3*pycaret_preds + 0.1*text_scores
        labels = final_scores > np.quantile(final_scores, 1 - self.contamination)
        
        return final_scores, labels, self._generate_explanations(df, labels)

    def _calculate_modified_zscore(self, df):
        """Robust z-score using median and MAD"""
        z_scores = pd.DataFrame()
        for col in self.schema['criteria_columns']:
            median = np.median(df[col])
            mad_val = mad(df[col])
            z_scores[col] = 0.6745 * (df[col] - median) / mad_val
        return z_scores.values

    def _percentile_anomalies(self, df):
        """Non-parametric range detection"""
        flags = np.zeros(len(df))
        for col in self.schema['criteria_columns']:
            lower = np.percentile(df[col], 5)
            upper = np.percentile(df[col], 95)
            flags |= (df[col] < lower) | (df[col] > upper)
        return flags

    def _pycaret_detection(self, df):
        """Automated model selection with PyCaret"""
        try:
            exp = setup(df[self.schema['criteria_columns']], 
                       silent=True, verbose=False)
            model = create_model('iforest', fraction=self.contamination)
            results = assign_model(model)
            return results['Anomaly_Score'].values
        except:
            return np.zeros(len(df))

    def _ensemble_detection(self, X):
        """Ensemble of multiple detection models"""
        scores = np.zeros(X.shape[0])
        for name, model in self.models.items():
            try:
                if hasattr(model, 'fit_predict'):
                    preds = model.fit_predict(X)
                else:
                    model.fit(X)
                    preds = model.decision_function(X)
                scores += self._normalize_scores(preds)
            except:
                continue
        return scores / len(self.models)

    def _analyze_comments(self, df):
        """Semantic analysis of comment columns"""
        comments = df[self.schema['comment_columns'][0]].fillna('')
        embeddings = self.text_encoder.encode(comments)
        
        # Detect outliers in semantic space
        umap_emb = UMAP().fit_transform(embeddings)
        lof = LocalOutlierFactor(contamination=self.contamination)
        return self._normalize_scores(lof.fit_predict(umap_emb))

    def _generate_explanations(self, df, labels):
        """Explainable AI component"""
        explanations = []
        for idx in np.where(labels)[0]:
            reasons = []
            record = df.iloc[idx]
            
            # Numeric criteria checks
            for col in self.schema['criteria_columns']:
                value = record[col]
                if value < np.percentile(df[col], 5):
                    reasons.append(f"{col} below 5th percentile ({value:.2f})")
                elif value > np.percentile(df[col], 95):
                    reasons.append(f"{col} above 95th percentile ({value:.2f})")
            
            # Textual explanations
            if self.schema.get('comment_columns'):
                comment = record[self.schema['comment_columns'][0]]
                if pd.notna(comment) and len(comment) > 0:
                    reasons.append(f"Unusual comment: '{comment[:50]}...'")
            
            explanations.append(" | ".join(reasons) if reasons else "Unknown pattern")
        
        return explanations

    @staticmethod
    def _normalize_scores(scores):
        """Normalize scores to 0-1 range"""
        return (scores - np.min(scores)) / (np.max(scores) - np.min(scores) + 1e-8)

class TextEncoderTransformer(TransformerMixin, BaseEstimator):
    """Custom transformer for text columns"""
    
    def __init__(self, encoder):
        self.encoder = encoder
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # Convert DataFrame to NumPy array if necessary
        if hasattr(X, 'values'):
            X = X.values
        # If the data is two-dimensional (150, 1), flatten it to (150,)
        if X.ndim == 2:
            X = X.ravel()
        # Encode each text entry row-wise
        res = np.array([self.encoder.encode(text) for text in X])
        return res

# Example Usage
if __name__ == "__main__":
    # Sample schema from LLM analysis
    schema = {
        'key_columns': ['Account', 'Company'],
        'criteria_columns': ['Gt_Balance', 'Hub_Balance'],
        'date_columns': ['As_of_Date'],
        'comment_columns': ['Notes']
    }
    
    # Generate sample data
    data = {
        'Account': np.random.randint(1000, 9999, 150),
        'Company': np.random.choice(['A', 'B', 'C'], 150),
        'Gt_Balance': np.concatenate([np.random.normal(10000, 500, 145), np.random.normal(50000, 1000, 5)]),
        'Hub_Balance': np.concatenate([np.random.normal(10000, 500, 145), np.random.normal(20000, 1000, 5)]),
        'As_of_Date': pd.date_range('2023-01-01', periods=150, freq='D'),
        'Notes': ['NAN']*145 + ['System error detected']*5
    }
    df = pd.DataFrame(data)

    # Initialize and run detector
    detector = AdaptiveAnomalyDetector(schema)
    scores, labels, explanations = detector.detect_anomalies(df)
    
    # Display results
    results = pd.DataFrame({
        'Account': df['Account'],
        'Anomaly_Score': scores,
        'Is_Anomaly': labels,
        'Explanation': explanations
    })
    
    print("\nDetection Results:")
    print(results[results['Is_Anomaly']].sort_values('Anomaly_Score', ascending=False))