import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope

class AnomalyDetector:
    def __init__(self, method='isolation_forest', contamination=0.1):
        self.method = method
        self.contamination = contamination
        self.detector = None
        
    def fit_detect(self, data):
        """Fit the detector and identify anomalies"""
        if self.method == 'isolation_forest':
            self.detector = IsolationForest(contamination=self.contamination, random_state=42)
        else:
            self.detector = EllipticEnvelope(contamination=self.contamination, random_state=42)
            
        # Fit and predict
        predictions = self.detector.fit_predict(data)
        
        # Return indices of normal samples (1) and anomalies (-1)
        normal_samples = data[predictions == 1]
        anomalies = data[predictions == -1]
        
        return {
            'normal_samples': normal_samples,
            'anomalies': anomalies,
            'anomaly_percentage': (len(anomalies) / len(data)) * 100
        }
    
    def check_data_quality(self, data):
        """Perform various data quality checks"""
        quality_report = {
            'total_samples': len(data),
            'missing_values': data.isnull().sum().to_dict(),
            'missing_percentage': (data.isnull().sum() / len(data) * 100).to_dict(),
            'zero_variance_columns': data.columns[data.var() == 0].tolist(),
            'correlation_matrix': data.corr().round(2).to_dict()
        }
        
        # Check for highly correlated features
        corr_matrix = data.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        high_corr_pairs = [(upper.index[i], upper.columns[j], upper.iloc[i,j])
                          for i, j in zip(*np.where(upper > 0.95))]
        
        quality_report['high_correlations'] = high_corr_pairs
        
        return quality_report
    
    def detect(self, data):
        """Main method to perform anomaly detection and data quality checks"""
        # Perform data quality checks
        quality_report = self.check_data_quality(data)
        
        # Detect anomalies
        anomaly_report = self.fit_detect(data)
        
        return {
            'quality_report': quality_report,
            'anomaly_report': anomaly_report,
            'is_data_valid': (anomaly_report['anomaly_percentage'] < 20 and  # Less than 20% anomalies
                            len(quality_report['zero_variance_columns']) == 0 and  # No zero variance columns
                            all(p < 30 for p in quality_report['missing_percentage'].values()))  # Less than 30% missing values
        } 