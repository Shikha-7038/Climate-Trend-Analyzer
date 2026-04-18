"""
Anomaly Detector Module - Identifies unusual climate patterns
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from scipy import stats

class ClimateAnomalyDetector:
    """
    Detects anomalies in climate data (extreme events, unusual patterns)
    """
    
    def __init__(self, data):
        """
        Initialize with climate data
        """
        self.data = data.copy()
        self.anomalies = {}
        
        if 'Date' in self.data.columns:
            self.data['Date'] = pd.to_datetime(self.data['Date'])
    
    def statistical_anomaly_detection(self, column, method='zscore', threshold=2.5):
        """
        Detect anomalies using statistical methods
        
        Methods:
        - 'zscore': Z-score method (values beyond threshold standard deviations)
        - 'iqr': Interquartile range method
        - 'modified_zscore': Robust z-score using median
        """
        series = self.data[column].dropna()
        anomaly_indices = []
        
        if method == 'zscore':
            z_scores = np.abs(stats.zscore(series))
            anomaly_indices = series[z_scores > threshold].index.tolist()
            self.anomalies[column] = {
                'method': 'zscore',
                'threshold': threshold,
                'anomaly_count': len(anomaly_indices),
                'anomaly_percentage': (len(anomaly_indices) / len(series)) * 100,
                'anomaly_indices': anomaly_indices,
                'anomaly_values': series[z_scores > threshold].to_dict()
            }
        
        elif method == 'iqr':
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            anomaly_indices = series[(series < lower_bound) | (series > upper_bound)].index.tolist()
            self.anomalies[column] = {
                'method': 'iqr',
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'anomaly_count': len(anomaly_indices),
                'anomaly_percentage': (len(anomaly_indices) / len(series)) * 100,
                'anomaly_indices': anomaly_indices,
                'anomaly_values': series[anomaly_indices].to_dict()
            }
        
        elif method == 'modified_zscore':
            median = series.median()
            mad = np.median(np.abs(series - median))
            modified_z_scores = 0.6745 * (series - median) / mad if mad != 0 else np.zeros(len(series))
            anomaly_indices = series[np.abs(modified_z_scores) > threshold].index.tolist()
            self.anomalies[column] = {
                'method': 'modified_zscore',
                'threshold': threshold,
                'anomaly_count': len(anomaly_indices),
                'anomaly_percentage': (len(anomaly_indices) / len(series)) * 100,
                'anomaly_indices': anomaly_indices,
                'anomaly_values': series[anomaly_indices].to_dict()
            }
        
        return self.anomalies[column]
    
    def isolation_forest_anomaly_detection(self, columns=None, contamination=0.05):
        """
        Detect anomalies using Isolation Forest algorithm
        
        Parameters:
        - columns: List of columns to use for detection
        - contamination: Expected proportion of anomalies (0-0.5)
        """
        if columns is None:
            columns = ['Temperature_C', 'Rainfall_mm', 'Humidity_Percent']
            columns = [col for col in columns if col in self.data.columns]
        
        # Prepare data
        X = self.data[columns].dropna()
        
        # Train Isolation Forest
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        predictions = iso_forest.fit_predict(X)
        
        # -1 indicates anomaly, 1 indicates normal
        anomaly_mask = predictions == -1
        
        self.anomalies['isolation_forest'] = {
            'method': 'isolation_forest',
            'columns_used': columns,
            'contamination': contamination,
            'anomaly_count': sum(anomaly_mask),
            'anomaly_percentage': (sum(anomaly_mask) / len(X)) * 100,
            'anomaly_indices': X[anomaly_mask].index.tolist()
        }
        
        return self.anomalies['isolation_forest']
    
    def extreme_event_detection(self, column, percentiles=[1, 99]):
        """
        Detect extreme events based on percentile thresholds
        
        Returns extreme low (below p1) and extreme high (above p99) events
        """
        series = self.data[column].dropna()
        
        p_low = percentiles[0]
        p_high = percentiles[1]
        
        low_threshold = np.percentile(series, p_low)
        high_threshold = np.percentile(series, p_high)
        
        extreme_lows = series[series < low_threshold]
        extreme_highs = series[series > high_threshold]
        
        result = {
            'column': column,
            'low_threshold': low_threshold,
            'high_threshold': high_threshold,
            'extreme_low_count': len(extreme_lows),
            'extreme_high_count': len(extreme_highs),
            'extreme_lows': extreme_lows.to_dict(),
            'extreme_highs': extreme_highs.to_dict(),
            'total_extreme_percentage': ((len(extreme_lows) + len(extreme_highs)) / len(series)) * 100
        }
        
        if 'extreme_events' not in self.anomalies:
            self.anomalies['extreme_events'] = {}
        self.anomalies['extreme_events'][column] = result
        
        return result
    
    def temporal_anomaly_detection(self, column, window=60, threshold=2):
        """
        Detect temporal anomalies using rolling statistics
        (e.g., values that deviate significantly from recent history)
        """
        series = self.data[column].dropna()
        
        # Calculate rolling mean and standard deviation
        rolling_mean = series.rolling(window=window, min_periods=window//2).mean()
        rolling_std = series.rolling(window=window, min_periods=window//2).std()
        
        # Calculate upper and lower bounds
        upper_bound = rolling_mean + (threshold * rolling_std)
        lower_bound = rolling_mean - (threshold * rolling_std)
        
        # Detect anomalies
        temporal_anomalies = series[(series > upper_bound) | (series < lower_bound)]
        
        result = {
            'column': column,
            'window': window,
            'threshold': threshold,
            'anomaly_count': len(temporal_anomalies),
            'anomaly_percentage': (len(temporal_anomalies) / len(series)) * 100,
            'anomalies': temporal_anomalies.to_dict(),
            'upper_bound_series': upper_bound.to_dict(),
            'lower_bound_series': lower_bound.to_dict()
        }
        
        if 'temporal_anomalies' not in self.anomalies:
            self.anomalies['temporal_anomalies'] = {}
        self.anomalies['temporal_anomalies'][column] = result
        
        return result
    
    def get_anomaly_summary(self):
        """
        Generate a summary of all detected anomalies
        """
        summary = {
            'total_anomalies_by_method': {},
            'combined_anomaly_dates': set(),
            'recommendations': []
        }
        
        # Collect all anomalies
        for key, value in self.anomalies.items():
            if isinstance(value, dict):
                if 'anomaly_count' in value:
                    summary['total_anomalies_by_method'][key] = value['anomaly_count']
                if 'anomaly_indices' in value:
                    summary['combined_anomaly_dates'].update(value['anomaly_indices'])
        
        # Add recommendations based on anomalies
        if 'extreme_events' in self.anomalies:
            for col, events in self.anomalies['extreme_events'].items():
                if events['extreme_high_count'] > 5:
                    summary['recommendations'].append(f"Multiple extreme high {col} events detected - investigate climate change impact")
                if events['extreme_low_count'] > 5:
                    summary['recommendations'].append(f"Multiple extreme low {col} events detected - unusual cooling patterns")
        
        if 'isolation_forest' in self.anomalies:
            if self.anomalies['isolation_forest']['anomaly_percentage'] > 10:
                summary['recommendations'].append("High percentage of anomalies detected - verify data quality or investigate climate regime shift")
        
        return summary
    
    def classify_anomaly_type(self, value, column):
        """
        Classify the type of anomaly for a given value
        """
        series = self.data[column].dropna()
        mean = series.mean()
        std = series.std()
        z_score = (value - mean) / std
        
        if abs(z_score) < 2:
            return "Normal"
        elif 2 <= abs(z_score) < 3:
            return "Mild Anomaly"
        elif 3 <= abs(z_score) < 4:
            return "Moderate Anomaly"
        else:
            return "Extreme Anomaly"

# Test the module
if __name__ == "__main__":
    from data_loader import ClimateDataLoader
    
    # Load data with anomalies
    loader = ClimateDataLoader()
    df = loader.generate_synthetic_climate_data()
    df = loader.add_anomalies(anomaly_percentage=3)
    
    # Detect anomalies
    detector = ClimateAnomalyDetector(df)
    
    # Statistical anomaly detection
    temp_anomalies = detector.statistical_anomaly_detection('Temperature_C', method='zscore', threshold=2.5)
    print(f"Temperature anomalies detected: {temp_anomalies['anomaly_count']}")
    
    # Extreme event detection
    extreme_events = detector.extreme_event_detection('Temperature_C', percentiles=[1, 99])
    print(f"Extreme heat events: {extreme_events['extreme_high_count']}")
    
    # Isolation Forest
    forest_anomalies = detector.isolation_forest_anomaly_detection()
    print(f"Isolation Forest anomalies: {forest_anomalies['anomaly_count']}")
    
    # Summary
    summary = detector.get_anomaly_summary()
    print(f"\nAnomaly Summary: {summary}")