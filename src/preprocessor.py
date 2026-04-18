"""
Data Preprocessor Module - Cleans and prepares climate data
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class ClimatePreprocessor:
    """
    Handles data cleaning, missing value imputation, and feature engineering
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.original_data = None
        self.processed_data = None
    
    def load_data(self, data):
        """
        Load data for preprocessing
        """
        self.original_data = data.copy()
        self.processed_data = data.copy()
        print(f"Loaded {len(self.processed_data)} records for preprocessing")
        return self
    
    def handle_missing_values(self, strategy='ffill'):
        """
        Handle missing values in the dataset
        
        Strategies:
        - 'ffill': Forward fill
        - 'bfill': Backward fill
        - 'mean': Fill with column mean
        - 'median': Fill with column median
        - 'drop': Drop rows with missing values
        """
        if self.processed_data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        missing_before = self.processed_data.isnull().sum().sum()
        
        if strategy == 'ffill':
            self.processed_data = self.processed_data.fillna(method='ffill')
        elif strategy == 'bfill':
            self.processed_data = self.processed_data.fillna(method='bfill')
        elif strategy == 'mean':
            for col in self.processed_data.select_dtypes(include=[np.number]).columns:
                self.processed_data[col] = self.processed_data[col].fillna(self.processed_data[col].mean())
        elif strategy == 'median':
            for col in self.processed_data.select_dtypes(include=[np.number]).columns:
                self.processed_data[col] = self.processed_data[col].fillna(self.processed_data[col].median())
        elif strategy == 'drop':
            self.processed_data = self.processed_data.dropna()
        
        missing_after = self.processed_data.isnull().sum().sum()
        print(f"Missing values: {missing_before} → {missing_after}")
        
        return self
    
    def remove_outliers(self, column, method='iqr', threshold=3):
        """
        Remove outliers from specified column
        
        Methods:
        - 'iqr': Interquartile range method (1.5 * IQR)
        - 'zscore': Z-score method (|z| > threshold)
        """
        if self.processed_data is None:
            raise ValueError("No data loaded.")
        
        initial_count = len(self.processed_data)
        
        if method == 'iqr':
            Q1 = self.processed_data[column].quantile(0.25)
            Q3 = self.processed_data[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            self.processed_data = self.processed_data[
                (self.processed_data[column] >= lower_bound) & 
                (self.processed_data[column] <= upper_bound)
            ]
        
        elif method == 'zscore':
            mean = self.processed_data[column].mean()
            std = self.processed_data[column].std()
            z_scores = np.abs((self.processed_data[column] - mean) / std)
            self.processed_data = self.processed_data[z_scores < threshold]
        
        removed = initial_count - len(self.processed_data)
        print(f"Removed {removed} outliers from column '{column}'")
        
        return self
    
    def create_features(self):
        """
        Create additional features for better analysis
        """
        if self.processed_data is None:
            raise ValueError("No data loaded.")
        
        # Ensure Date is datetime
        if 'Date' in self.processed_data.columns:
            self.processed_data['Date'] = pd.to_datetime(self.processed_data['Date'])
            self.processed_data['Year'] = self.processed_data['Date'].dt.year
            self.processed_data['Month'] = self.processed_data['Date'].dt.month
            self.processed_data['Season'] = self.processed_data['Month'].map({
                12: 'Winter', 1: 'Winter', 2: 'Winter',
                3: 'Spring', 4: 'Spring', 5: 'Spring',
                6: 'Summer', 7: 'Summer', 8: 'Summer',
                9: 'Fall', 10: 'Fall', 11: 'Fall'
            })
        
        # Rolling averages (smoothing)
        if 'Temperature_C' in self.processed_data.columns:
            self.processed_data['Temp_5yr_Avg'] = self.processed_data['Temperature_C'].rolling(window=60).mean()
            self.processed_data['Temp_10yr_Avg'] = self.processed_data['Temperature_C'].rolling(window=120).mean()
        
        if 'Rainfall_mm' in self.processed_data.columns:
            self.processed_data['Rainfall_5yr_Avg'] = self.processed_data['Rainfall_mm'].rolling(window=60).mean()
        
        # Year-over-year changes
        if 'Temperature_C' in self.processed_data.columns:
            self.processed_data['Temp_Change_1yr'] = self.processed_data['Temperature_C'].diff(12)
        
        print(f"Created additional features. New columns: {list(self.processed_data.columns)}")
        return self
    
    def normalize_data(self, columns=None, method='standard'):
        """
        Normalize specified columns
        
        Methods:
        - 'standard': StandardScaler (mean=0, std=1)
        - 'minmax': MinMaxScaler (range 0-1)
        """
        if self.processed_data is None:
            raise ValueError("No data loaded.")
        
        if columns is None:
            columns = self.processed_data.select_dtypes(include=[np.number]).columns
            columns = [col for col in columns if col not in ['Year', 'Month']]
        
        for col in columns:
            if method == 'standard':
                self.processed_data[f'{col}_normalized'] = (self.processed_data[col] - self.processed_data[col].mean()) / self.processed_data[col].std()
            elif method == 'minmax':
                min_val = self.processed_data[col].min()
                max_val = self.processed_data[col].max()
                self.processed_data[f'{col}_normalized'] = (self.processed_data[col] - min_val) / (max_val - min_val)
        
        print(f"Normalized {len(columns)} columns using {method} method")
        return self
    
    def get_processed_data(self):
        """
        Returns the processed data
        """
        return self.processed_data
    
    def save_processed_data(self, filepath):
        """
        Save processed data to CSV
        """
        if self.processed_data is not None:
            self.processed_data.to_csv(filepath, index=False)
            print(f"Saved processed data to {filepath}")
    
    def get_preprocessing_report(self):
        """
        Generate a preprocessing report
        """
        if self.processed_data is None:
            return "No data processed yet."
        
        report = {
            'original_shape': self.original_data.shape if self.original_data is not None else None,
            'processed_shape': self.processed_data.shape,
            'missing_values_remaining': self.processed_data.isnull().sum().sum(),
            'data_types': self.processed_data.dtypes.to_dict(),
            'memory_usage': self.processed_data.memory_usage(deep=True).sum() / 1024**2  # MB
        }
        
        return report

# Test the module
if __name__ == "__main__":
    # Create sample data
    from data_loader import ClimateDataLoader
    loader = ClimateDataLoader()
    df = loader.generate_synthetic_climate_data()
    
    # Preprocess
    preprocessor = ClimatePreprocessor()
    preprocessor.load_data(df)
    preprocessor.handle_missing_values()
    preprocessor.remove_outliers('Temperature_C')
    preprocessor.create_features()
    
    print("\nPreprocessing Report:")
    print(preprocessor.get_preprocessing_report())
    
    processed_df = preprocessor.get_processed_data()
    print(f"\nProcessed data shape: {processed_df.shape}")