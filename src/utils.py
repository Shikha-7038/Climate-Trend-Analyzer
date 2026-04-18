"""
Utils Module - Helper functions for the Climate Trend Analyzer project
"""
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def setup_logging(log_file='logs/project.log'):
    """
    Set up logging configuration
    
    Parameters:
    - log_file: Path to log file
    
    Returns:
    - logger: Configured logger object
    """
    import logging
    
    # Create logs directory if it doesn't exist
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

def save_metadata(data_info, filepath='outputs/reports/metadata.json'):
    """
    Save metadata about the dataset and analysis
    
    Parameters:
    - data_info: Dictionary containing metadata
    - filepath: Path to save JSON file
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    metadata = {
        'created_at': datetime.now().isoformat(),
        'data_info': data_info
    }
    
    with open(filepath, 'w') as f:
        json.dump(metadata, f, indent=4, default=str)
    
    print(f"Metadata saved to {filepath}")
    return metadata

def load_metadata(filepath='outputs/reports/metadata.json'):
    """
    Load metadata from JSON file
    
    Parameters:
    - filepath: Path to JSON file
    
    Returns:
    - metadata: Dictionary containing metadata
    """
    try:
        with open(filepath, 'r') as f:
            metadata = json.load(f)
        print(f"Metadata loaded from {filepath}")
        return metadata
    except FileNotFoundError:
        print(f"No metadata file found at {filepath}")
        return None

def validate_data(df, required_columns=None):
    """
    Validate that the dataframe has required columns and no critical issues
    
    Parameters:
    - df: Pandas DataFrame
    - required_columns: List of required column names
    
    Returns:
    - is_valid: Boolean indicating if data is valid
    - issues: List of issues found
    """
    issues = []
    
    if df is None:
        issues.append("DataFrame is None")
        return False, issues
    
    if len(df) == 0:
        issues.append("DataFrame is empty")
        return False, issues
    
    # Check required columns
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            issues.append(f"Missing required columns: {missing_cols}")
    
    # Check for excessive missing values
    missing_percentages = (df.isnull().sum() / len(df)) * 100
    high_missing = missing_percentages[missing_percentages > 50].to_dict()
    if high_missing:
        issues.append(f"Columns with >50% missing values: {high_missing}")
    
    # Check for duplicate rows
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        issues.append(f"Found {duplicates} duplicate rows")
    
    is_valid = len(issues) == 0
    return is_valid, issues

def calculate_statistics(df, columns=None):
    """
    Calculate comprehensive statistics for specified columns
    
    Parameters:
    - df: Pandas DataFrame
    - columns: List of columns to analyze (default: all numeric columns)
    
    Returns:
    - stats: Dictionary containing statistics
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    stats = {}
    
    for col in columns:
        if col in df.columns:
            series = df[col].dropna()
            stats[col] = {
                'count': len(series),
                'mean': series.mean(),
                'median': series.median(),
                'std': series.std(),
                'min': series.min(),
                'max': series.max(),
                'q1': series.quantile(0.25),
                'q3': series.quantile(0.75),
                'iqr': series.quantile(0.75) - series.quantile(0.25),
                'skewness': series.skew(),
                'kurtosis': series.kurtosis()
            }
    
    return stats

def detect_seasonality(df, date_column='Date', value_column='Temperature_C'):
    """
    Detect seasonality in time series data using autocorrelation
    
    Parameters:
    - df: Pandas DataFrame with time series data
    - date_column: Name of date column
    - value_column: Name of value column to analyze
    
    Returns:
    - seasonality_info: Dictionary with seasonality detection results
    """
    from statsmodels.tsa.stattools import acf
    
    # Ensure data is sorted by date
    if date_column in df.columns:
        df_sorted = df.sort_values(date_column)
        values = df_sorted[value_column].dropna().values
    else:
        values = df[value_column].dropna().values
    
    # Calculate autocorrelation
    try:
        autocorr = acf(values, nlags=24, fft=False)
        
        # Look for peaks at lags 12 (yearly) and 6 (half-yearly)
        yearly_peak = autocorr[12] if len(autocorr) > 12 else 0
        half_yearly_peak = autocorr[6] if len(autocorr) > 6 else 0
        
        has_seasonality = yearly_peak > 0.3 or half_yearly_peak > 0.3
        
        seasonality_info = {
            'has_seasonality': has_seasonality,
            'yearly_autocorrelation': yearly_peak,
            'half_yearly_autocorrelation': half_yearly_peak,
            'autocorrelation_values': autocorr.tolist()[:24] if len(autocorr) >= 24 else autocorr.tolist()
        }
    except Exception as e:
        seasonality_info = {
            'has_seasonality': False,
            'error': str(e)
        }
    
    return seasonality_info

def export_to_csv(df, filepath, index=False):
    """
    Export dataframe to CSV with proper error handling
    
    Parameters:
    - df: Pandas DataFrame
    - filepath: Path to save CSV
    - index: Whether to include index
    
    Returns:
    - success: Boolean indicating success
    """
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        df.to_csv(filepath, index=index)
        print(f"Data exported to {filepath}")
        return True
    except Exception as e:
        print(f"Error exporting to CSV: {e}")
        return False

def export_to_excel(df, filepath, sheet_name='Climate_Data'):
    """
    Export dataframe to Excel with proper error handling
    
    Parameters:
    - df: Pandas DataFrame
    - filepath: Path to save Excel
    - sheet_name: Name of the sheet
    
    Returns:
    - success: Boolean indicating success
    """
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        df.to_excel(filepath, sheet_name=sheet_name, index=False)
        print(f"Data exported to {filepath}")
        return True
    except Exception as e:
        print(f"Error exporting to Excel: {e}")
        return False

def create_summary_table(df, columns=None):
    """
    Create a formatted summary table for reporting
    
    Parameters:
    - df: Pandas DataFrame
    - columns: List of columns to summarize
    
    Returns:
    - summary_df: Formatted summary DataFrame
    """
    if columns is None:
        columns = ['Temperature_C', 'Rainfall_mm', 'Humidity_Percent', 'CO2_ppm']
        columns = [col for col in columns if col in df.columns]
    
    summary_data = []
    
    for col in columns:
        series = df[col].dropna()
        summary_data.append({
            'Variable': col.replace('_', ' '),
            'Mean': f"{series.mean():.2f}",
            'Median': f"{series.median():.2f}",
            'Std Dev': f"{series.std():.2f}",
            'Min': f"{series.min():.2f}",
            'Max': f"{series.max():.2f}",
            '5th Percentile': f"{series.quantile(0.05):.2f}",
            '95th Percentile': f"{series.quantile(0.95):.2f}"
        })
    
    summary_df = pd.DataFrame(summary_data)
    return summary_df

def generate_time_features(df, date_column='Date'):
    """
    Generate time-based features from date column
    
    Parameters:
    - df: Pandas DataFrame
    - date_column: Name of date column
    
    Returns:
    - df: DataFrame with new time features
    """
    if date_column not in df.columns:
        print(f"Column '{date_column}' not found")
        return df
    
    # Ensure datetime
    df[date_column] = pd.to_datetime(df[date_column])
    
    # Extract time features
    df['Year'] = df[date_column].dt.year
    df['Month'] = df[date_column].dt.month
    df['Day'] = df[date_column].dt.day
    df['DayOfYear'] = df[date_column].dt.dayofyear
    df['WeekOfYear'] = df[date_column].dt.isocalendar().week.astype(int)
    df['Quarter'] = df[date_column].dt.quarter
    
    # Cyclical features for seasonality
    df['Month_Sin'] = np.sin(2 * np.pi * df['Month'] / 12)
    df['Month_Cos'] = np.cos(2 * np.pi * df['Month'] / 12)
    df['DayOfYear_Sin'] = np.sin(2 * np.pi * df['DayOfYear'] / 365)
    df['DayOfYear_Cos'] = np.cos(2 * np.pi * df['DayOfYear'] / 365)
    
    # Is weekend?
    df['IsWeekend'] = (df[date_column].dt.dayofweek >= 5).astype(int)
    
    return df

def detect_data_drift(train_df, test_df, columns=None, threshold=0.1):
    """
    Detect data drift between training and test datasets
    
    Parameters:
    - train_df: Training data DataFrame
    - test_df: Test data DataFrame
    - columns: Columns to check for drift
    - threshold: Maximum allowed difference in mean (as multiple of std)
    
    Returns:
    - drift_report: Dictionary with drift detection results
    """
    if columns is None:
        columns = train_df.select_dtypes(include=[np.number]).columns.tolist()
        columns = [col for col in columns if col in test_df.columns]
    
    drift_report = {
        'columns_checked': columns,
        'drift_detected': False,
        'drifts': {}
    }
    
    for col in columns:
        train_mean = train_df[col].mean()
        test_mean = test_df[col].mean()
        train_std = train_df[col].std()
        
        if train_std > 0:
            drift_magnitude = abs(test_mean - train_mean) / train_std
            has_drift = drift_magnitude > threshold
            
            drift_report['drifts'][col] = {
                'train_mean': train_mean,
                'test_mean': test_mean,
                'drift_magnitude': drift_magnitude,
                'has_drift': has_drift
            }
            
            if has_drift:
                drift_report['drift_detected'] = True
    
    return drift_report

def get_memory_usage(df):
    """
    Get memory usage of DataFrame
    
    Parameters:
    - df: Pandas DataFrame
    
    Returns:
    - memory_info: Dictionary with memory usage information
    """
    memory_bytes = df.memory_usage(deep=True).sum()
    memory_mb = memory_bytes / (1024 ** 2)
    memory_gb = memory_bytes / (1024 ** 3)
    
    # Per column memory
    per_column = df.memory_usage(deep=True)
    
    return {
        'total_bytes': memory_bytes,
        'total_mb': round(memory_mb, 2),
        'total_gb': round(memory_gb, 2),
        'per_column': per_column.to_dict()
    }

def optimize_dtypes(df):
    """
    Optimize data types to reduce memory usage
    
    Parameters:
    - df: Pandas DataFrame
    
    Returns:
    - df_optimized: DataFrame with optimized dtypes
    """
    df_optimized = df.copy()
    
    for col in df_optimized.columns:
        col_type = df_optimized[col].dtype
        
        if col_type != 'object':
            # Integer optimization
            if col_type == 'int64':
                c_min = df_optimized[col].min()
                c_max = df_optimized[col].max()
                
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df_optimized[col] = df_optimized[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df_optimized[col] = df_optimized[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df_optimized[col] = df_optimized[col].astype(np.int32)
            
            # Float optimization
            elif col_type == 'float64':
                df_optimized[col] = df_optimized[col].astype(np.float32)
    
    return df_optimized

def print_progress_bar(iteration, total, prefix='Progress:', suffix='', decimals=1, length=50, fill='█'):
    """
    Print a progress bar in console
    
    Parameters:
    - iteration: Current iteration
    - total: Total iterations
    - prefix: Prefix string
    - suffix: Suffix string
    - decimals: Decimal places for percentage
    - length: Character length of bar
    - fill: Bar fill character
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='')
    
    if iteration == total:
        print()

def format_number(num):
    """
    Format large numbers with K, M, B suffixes
    
    Parameters:
    - num: Number to format
    
    Returns:
    - formatted: Formatted string
    """
    if abs(num) >= 1_000_000_000:
        return f"{num/1_000_000_000:.1f}B"
    elif abs(num) >= 1_000_000:
        return f"{num/1_000_000:.1f}M"
    elif abs(num) >= 1_000:
        return f"{num/1_000:.1f}K"
    else:
        return str(num)

def safe_divide(a, b, default=0):
    """
    Safely divide two numbers, returning default if division by zero
    
    Parameters:
    - a: Numerator
    - b: Denominator
    - default: Value to return if b is zero
    
    Returns:
    - result: a/b or default
    """
    try:
        return a / b if b != 0 else default
    except (TypeError, ZeroDivisionError):
        return default

def get_file_size(filepath):
    """
    Get file size in human-readable format
    
    Parameters:
    - filepath: Path to file
    
    Returns:
    - size_str: Human-readable file size
    """
    if not os.path.exists(filepath):
        return "File not found"
    
    size_bytes = os.path.getsize(filepath)
    
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    
    return f"{size_bytes:.1f} TB"

def create_directory_structure(base_path):
    """
    Create the complete project directory structure
    
    Parameters:
    - base_path: Base path for the project
    
    Returns:
    - created_dirs: List of created directories
    """
    directories = [
        f"{base_path}/data/raw",
        f"{base_path}/data/processed",
        f"{base_path}/notebooks",
        f"{base_path}/src",
        f"{base_path}/models",
        f"{base_path}/outputs/figures",
        f"{base_path}/outputs/reports",
        f"{base_path}/images",
        f"{base_path}/docs",
        f"{base_path}/app",
        f"{base_path}/logs"
    ]
    
    created = []
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        created.append(directory)
    
    print(f"Created {len(created)} directories")
    return created

# Test the module
if __name__ == "__main__":
    # Test utility functions
    print("Testing Utils Module...")
    
    # Test number formatting
    print(f"1,000,000 formatted: {format_number(1000000)}")
    print(f"1,500,000,000 formatted: {format_number(1500000000)}")
    
    # Test safe division
    print(f"10/2 = {safe_divide(10, 2)}")
    print(f"10/0 = {safe_divide(10, 0)}")
    
    # Test progress bar
    print("\nTesting progress bar:")
    for i in range(101):
        print_progress_bar(i, 100, prefix='Test:', suffix='Complete')
    
    print("\nUtils module ready!")