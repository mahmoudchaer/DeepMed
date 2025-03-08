from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
import logging
import sys
import os

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

app = Flask(__name__)

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
            'anomaly_percentage': (len(anomalies) / len(data)) * 100,
            'anomaly_indices': np.where(predictions == -1)[0].tolist()
        }
    
    def check_data_quality(self, data):
        """Perform various data quality checks"""
        quality_report = {
            'total_samples': len(data),
            'missing_values': data.isnull().sum().to_dict(),
            'missing_percentage': (data.isnull().sum() / len(data) * 100).to_dict(),
            'zero_variance_columns': data.columns[data.var() == 0].tolist()
        }
        
        # Check for highly correlated features
        try:
            corr_matrix = data.corr().abs()
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            high_corr_pairs = []
            for i, j in zip(*np.where(upper > 0.95)):
                high_corr_pairs.append([upper.index[i], upper.columns[j], float(upper.iloc[i,j])])
            
            quality_report['high_correlations'] = high_corr_pairs
        except Exception as e:
            quality_report['high_correlations'] = []
            logging.warning(f"Error computing correlations: {str(e)}")
        
        return quality_report
    
    def detect(self, data):
        """Main method to perform anomaly detection and data quality checks"""
        # Perform data quality checks
        quality_report = self.check_data_quality(data)
        
        # Detect anomalies
        anomaly_report = self.fit_detect(data)
        
        # Convert pandas DataFrames to dictionaries for JSON serialization
        anomaly_report['normal_samples'] = anomaly_report['normal_samples'].to_dict(orient='records')
        anomaly_report['anomalies'] = anomaly_report['anomalies'].to_dict(orient='records')
        
        return {
            'quality_report': quality_report,
            'anomaly_report': anomaly_report,
            'is_data_valid': (anomaly_report['anomaly_percentage'] < 20 and  # Less than 20% anomalies
                            len(quality_report['zero_variance_columns']) == 0 and  # No zero variance columns
                            all(p < 30 for p in quality_report['missing_percentage'].values()))  # Less than 30% missing values
        }

# Create a global AnomalyDetector instance
anomaly_detector = AnomalyDetector()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "service": "anomaly_detector_api"})

@app.route('/detect_anomalies', methods=['POST'])
def detect_anomalies():
    """
    Anomaly detection endpoint
    
    Expected JSON input:
    {
        "data": {...},  # Data in JSON format that can be loaded into a pandas DataFrame
        "method": "isolation_forest" or "elliptic_envelope",  # Optional method to use (default: "isolation_forest")
        "contamination": 0.1  # Optional contamination parameter (default: 0.1)
    }
    
    Returns:
    {
        "is_data_valid": true/false,  # Whether the data passes quality checks
        "quality_report": {...},  # Report on data quality issues
        "anomaly_report": {...},  # Report on detected anomalies
        "message": "Anomaly detection completed successfully"
    }
    """
    try:
        # Get request data
        request_data = request.json
        
        if not request_data or 'data' not in request_data:
            return jsonify({"error": "Invalid request. Missing 'data'"}), 400
        
        # Convert JSON to DataFrame
        try:
            data = pd.DataFrame.from_dict(request_data['data'])
            
            # Set method if provided
            if 'method' in request_data and request_data['method']:
                anomaly_detector.method = request_data['method']
                
            # Set contamination if provided
            if 'contamination' in request_data and request_data['contamination']:
                anomaly_detector.contamination = float(request_data['contamination'])
                
        except Exception as e:
            return jsonify({"error": f"Failed to convert JSON to DataFrame: {str(e)}"}), 400
        
        # Detect anomalies
        results = anomaly_detector.detect(data)
        
        # Return detection results
        return jsonify({
            "is_data_valid": results['is_data_valid'],
            "quality_report": results['quality_report'],
            "anomaly_report": results['anomaly_report'],
            "message": "Anomaly detection completed successfully"
        })
    
    except Exception as e:
        logging.error(f"Error in detect_anomalies endpoint: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/quality_check', methods=['POST'])
def quality_check():
    """
    Data quality check endpoint (without anomaly detection)
    
    Expected JSON input:
    {
        "data": {...}  # Data in JSON format that can be loaded into a pandas DataFrame
    }
    
    Returns:
    {
        "quality_report": {...},  # Report on data quality issues
        "message": "Quality check completed successfully"
    }
    """
    try:
        # Get request data
        request_data = request.json
        
        if not request_data or 'data' not in request_data:
            return jsonify({"error": "Invalid request. Missing 'data'"}), 400
        
        # Convert JSON to DataFrame
        try:
            data = pd.DataFrame.from_dict(request_data['data'])
        except Exception as e:
            return jsonify({"error": f"Failed to convert JSON to DataFrame: {str(e)}"}), 400
        
        # Check data quality
        quality_report = anomaly_detector.check_data_quality(data)
        
        # Return quality report
        return jsonify({
            "quality_report": quality_report,
            "message": "Quality check completed successfully"
        })
    
    except Exception as e:
        logging.error(f"Error in quality_check endpoint: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Run the app on port 5003
    port = int(os.environ.get('PORT', 5003))
    app.run(host='0.0.0.0', port=port) 