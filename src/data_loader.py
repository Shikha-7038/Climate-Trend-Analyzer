"""
Data Loader Module - Handles loading climate datasets
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class ClimateDataLoader:
    """
    Loads climate data from various sources or generates synthetic data
    """
    
    def __init__(self):
        self.data = None
    
    def generate_synthetic_climate_data(self, start_year=1900, end_year=2024):
        """
        Generates realistic synthetic climate data for demonstration
        
        Parameters:
        - start_year: Starting year for data
        - end_year: Ending year for data
        
        Returns:
        - DataFrame with climate data
        """
        print(f"Generating synthetic climate data from {start_year} to {end_year}...")
        
        # Create date range
        dates = pd.date_range(start=f"{start_year}-01-01", 
                              end=f"{end_year}-12-31", 
                              freq='M')
        
        n_years = end_year - start_year + 1
        n_points = len(dates)
        
        # Base temperature with global warming trend (0.018°C per year increase)
        years = np.array([date.year for date in dates])
        base_temp = 14.0  # Base global temperature in °C
        
        # Long-term warming trend (accelerating slightly)
        warming_trend = 0.018 * (years - start_year)
        # Add acceleration in recent decades
        acceleration = np.where(years > 1980, 0.0005 * (years - 1980) ** 2, 0)
        
        # Seasonal variation (sinusoidal pattern)
        months = np.array([date.month for date in dates])
        seasonal = -5 * np.cos(2 * np.pi * (months - 1) / 12)
        
        # Random noise
        noise = np.random.normal(0, 0.5, n_points)
        
        # Calculate temperature
        temperature = base_temp + warming_trend + acceleration + seasonal + noise
        
        # Rainfall pattern (mm)
        # Decreasing trend in many regions, with seasonal variation
        base_rainfall = 80
        rainfall_trend = -0.05 * (years - start_year)  # slight decrease
        rainfall_trend = np.maximum(rainfall_trend, -30)  # cap the decrease
        
        # Seasonal rainfall (more in summer/wet season)
        rainfall_seasonal = 40 * np.sin(2 * np.pi * (months - 6) / 12)
        rainfall_noise = np.random.normal(0, 15, n_points)
        
        rainfall = base_rainfall + rainfall_trend + rainfall_seasonal + rainfall_noise
        rainfall = np.maximum(rainfall, 5)  # minimum 5mm
        
        # Humidity (%) - slight decreasing trend
        base_humidity = 65
        humidity_trend = -0.03 * (years - start_year)
        humidity_trend = np.maximum(humidity_trend, -15)
        humidity_seasonal = 15 * np.sin(2 * np.pi * (months - 3) / 12)
        humidity_noise = np.random.normal(0, 5, n_points)
        
        humidity = base_humidity + humidity_trend + humidity_seasonal + humidity_noise
        humidity = np.clip(humidity, 20, 95)
        
        # CO2 levels (ppm) - increasing trend
        base_co2 = 280
        co2_trend = 1.5 * ((years - start_year) / 100) ** 2  # accelerating
        co2_noise = np.random.normal(0, 2, n_points)
        co2_levels = base_co2 + co2_trend * 100 + co2_noise
        
        # Create DataFrame
        self.data = pd.DataFrame({
            'Date': dates,
            'Year': years,
            'Month': months,
            'Temperature_C': temperature,
            'Rainfall_mm': rainfall,
            'Humidity_Percent': humidity,
            'CO2_ppm': co2_levels
        })
        
        print(f"Generated {len(self.data)} records")
        return self.data
    
    def add_anomalies(self, anomaly_percentage=2):
        """
        Adds artificial anomalies to the data for demonstration
        """
        if self.data is None:
            raise ValueError("No data loaded. Generate or load data first.")
        
        n_anomalies = int(len(self.data) * anomaly_percentage / 100)
        anomaly_indices = np.random.choice(self.data.index, n_anomalies, replace=False)
        
        # Add temperature spikes (heat waves) or drops
        for idx in anomaly_indices:
            if np.random.random() > 0.5:
                # Heat wave - increase temperature by 3-6°C
                self.data.loc[idx, 'Temperature_C'] += np.random.uniform(3, 6)
            else:
                # Unusual cold - decrease temperature by 2-5°C
                self.data.loc[idx, 'Temperature_C'] -= np.random.uniform(2, 5)
        
        # Add extreme rainfall events
        n_rain_anomalies = n_anomalies // 2
        rain_anomaly_indices = np.random.choice(self.data.index, n_rain_anomalies, replace=False)
        for idx in rain_anomaly_indices:
            self.data.loc[idx, 'Rainfall_mm'] *= np.random.uniform(2, 4)
        
        print(f"Added {n_anomalies} temperature anomalies and {n_rain_anomalies} rainfall anomalies")
        return self.data
    
    def load_real_dataset(self, file_path):
        """
        Load a real climate dataset from CSV file
        """
        try:
            self.data = pd.read_csv(file_path)
            print(f"Loaded data from {file_path}")
            print(f"Shape: {self.data.shape}")
            return self.data
        except Exception as e:
            print(f"Error loading file: {e}")
            return None
    
    def get_summary(self):
        """
        Returns summary statistics of the data
        """
        if self.data is None:
            return None
        
        summary = {
            'shape': self.data.shape,
            'columns': list(self.data.columns),
            'date_range': f"{self.data['Date'].min()} to {self.data['Date'].max()}",
            'missing_values': self.data.isnull().sum().to_dict(),
            'statistics': self.data.describe().to_dict()
        }
        return summary

# Test the module
if __name__ == "__main__":
    loader = ClimateDataLoader()
    df = loader.generate_synthetic_climate_data()
    df = loader.add_anomalies(anomaly_percentage=2)
    print(loader.get_summary())
    print("\nFirst 5 rows:")
    print(df.head())