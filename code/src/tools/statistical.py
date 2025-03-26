import numpy as np
import pandas as pd
from typing import Dict, Any
from langchain_core.tools import tool

class StatisticalAnalysisTool:
    def __init__(self, historical_data: pd.DataFrame, current_data: pd.DataFrame):
        self.historical = historical_data
        self.current = current_data
        
        # Precompute historical metrics once
        self.hist_diff = self.historical['Bank Statement Amount'] - self.historical['Book Records Amount']
        self.mean_diff = np.mean(self.hist_diff)
        self.std_diff = np.std(self.hist_diff) if len(self.hist_diff) > 1 else 0
        self.q1, self.q3 = np.percentile(self.hist_diff, [25, 75]) if len(self.hist_diff) > 0 else (0, 0)
        self.iqr = self.q3 - self.q1

    @tool
    def statistical_analysis(self, transaction_id: str) -> Dict[str, Any]:
        """Perform comprehensive statistical analysis for transaction reconciliation"""
        try:
            # Get current transaction
            current_txn = self.current[self.current['Transaction ID'] == transaction_id].iloc[0]
            current_diff = current_txn['Bank Statement Amount'] - current_txn['Book Records Amount']
            # Calculate metrics
            z_score = (current_diff - self.mean_diff) / self.std_diff if self.std_diff != 0 else 0
            percentile = np.mean(self.hist_diff < current_diff) * 100  # Percentile rank
            
            # Outlier detection
            is_outlier = current_diff < (self.q1 - 1.5*self.iqr) or current_diff > (self.q3 + 1.5*self.iqr)
            
            # Trend analysis
            similar_txns = self.historical[
                (self.historical['Account Number'] == current_txn['Account Number']) &
                (self.historical['Bank Name'] == current_txn['Bank Name'])
            ]
            avg_similar_diff = similar_txns['Bank Statement Amount'].mean() - similar_txns['Book Records Amount'].mean()
            
            return {
                'transaction_id': transaction_id,
                'current_amount_difference': round(current_diff, 2),
                'z_score': round(z_score, 2),
                'percentile_rank': round(percentile, 1),
                'is_outlier': bool(is_outlier),
                'historical_comparison': {
                    'mean_difference': round(self.mean_diff, 2),
                    'std_deviation': round(self.std_diff, 2),
                    'iqr_range': [round(self.q1, 2), round(self.q3, 2)],
                    'similar_transactions_avg_diff': round(avg_similar_diff, 2)
                },
                'interpretation': self._generate_interpretation(current_diff, z_score, is_outlier)
            }
            
        except IndexError:
            return {'error': f'Transaction {transaction_id} not found'}
        except Exception as e:
            return {'error': f'Analysis failed: {str(e)}'}

    def _generate_interpretation(self, diff: float, z: float, outlier: bool) -> str:
        """Generate human-readable insights"""
        interpretations = []
        
        if abs(z) > 3:
            interpretations.append("Extreme deviation from historical norms (|z| > 3)")
        elif abs(z) > 2:
            interpretations.append("Significant deviation (|z| > 2)")
            
        if outlier:
            interpretations.append("IQR outlier detected")
            
        if not interpretations:
            interpretations.append("Within expected historical ranges")
            
        return f"Amount difference of ${diff:.2f}: " + ", ".join(interpretations)