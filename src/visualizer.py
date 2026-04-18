"""
Visualizer Module - Creates professional static visualizations
Saves images to outputs/figures/ for documentation and reports
"""
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import os

# Set style for better looking plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class ClimateVisualizer:
    """
    Creates comprehensive STATIC visualizations for climate analysis
    Saves images to disk for documentation and reports
    """
    
    def __init__(self, data):
        """
        Initialize with climate data
        """
        self.data = data.copy()
        if 'Date' in self.data.columns:
            self.data['Date'] = pd.to_datetime(self.data['Date'])
        
        # Create output directory if it doesn't exist
        os.makedirs('outputs/figures', exist_ok=True)
    
    def plot_temperature_trend(self, save_path='outputs/figures/temperature_trend.png'):
        """
        Plot global temperature trend over time with smoothing
        Saves as static PNG file
        """
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # Plot 1: Raw temperature data
        axes[0].plot(self.data['Date'], self.data['Temperature_C'], 
                     alpha=0.3, linewidth=0.5, label='Monthly Data', color='blue')
        
        # Add rolling averages
        if len(self.data) > 60:
            rolling_5yr = self.data['Temperature_C'].rolling(window=60).mean()
            axes[0].plot(self.data['Date'], rolling_5yr, 
                        linewidth=2, label='5-Year Average', color='red')
        
        axes[0].set_title('Global Temperature Trend (1900-2024)', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Year')
        axes[0].set_ylabel('Temperature (°C)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Temperature anomalies
        baseline_period = self.data[self.data['Date'].dt.year.between(1951, 1980)]['Temperature_C'].mean()
        self.data['Temperature_Anomaly'] = self.data['Temperature_C'] - baseline_period
        
        colors = ['red' if x > 0 else 'blue' for x in self.data['Temperature_Anomaly']]
        axes[1].bar(self.data['Date'], self.data['Temperature_Anomaly'], 
                    color=colors, alpha=0.7, width=20)
        axes[1].axhline(y=0, color='black', linestyle='-', linewidth=1)
        axes[1].set_title('Temperature Anomalies (Relative to 1951-1980 Baseline)', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Year')
        axes[1].set_ylabel('Temperature Anomaly (°C)')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved static image: {save_path}")
        return save_path
    
    def plot_seasonal_patterns(self, save_path='outputs/figures/seasonal_pattern.png'):
        """
        Plot seasonal patterns for different climate variables
        Saves as static PNG file
        """
        # Create seasonal data
        self.data['Month'] = self.data['Date'].dt.month
        self.data['Season'] = self.data['Month'].map({
            12: 'Winter', 1: 'Winter', 2: 'Winter',
            3: 'Spring', 4: 'Spring', 5: 'Spring',
            6: 'Summer', 7: 'Summer', 8: 'Summer',
            9: 'Fall', 10: 'Fall', 11: 'Fall'
        })
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Temperature by season
        seasonal_temp = self.data.groupby(['Year', 'Season'])['Temperature_C'].mean().reset_index()
        seasons_order = ['Winter', 'Spring', 'Summer', 'Fall']
        
        for season in seasons_order:
            season_data = seasonal_temp[seasonal_temp['Season'] == season]
            axes[0, 0].plot(season_data['Year'], season_data['Temperature_C'], 
                           label=season, linewidth=2)
        
        axes[0, 0].set_title('Temperature by Season (1900-2024)', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Year')
        axes[0, 0].set_ylabel('Temperature (°C)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Monthly temperature patterns
        monthly_data = []
        for month in range(1, 13):
            month_temps = self.data[self.data['Month'] == month]['Temperature_C']
            monthly_data.append(month_temps)
        
        axes[0, 1].boxplot(monthly_data, labels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
        axes[0, 1].set_title('Monthly Temperature Distribution', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Month')
        axes[0, 1].set_ylabel('Temperature (°C)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Rainfall by season
        seasonal_rainfall = self.data.groupby('Season')['Rainfall_mm'].mean()
        colors = ['lightblue', 'lightgreen', 'gold', 'orange']
        axes[1, 0].bar(seasonal_rainfall.index, seasonal_rainfall.values, color=colors)
        axes[1, 0].set_title('Average Rainfall by Season', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Season')
        axes[1, 0].set_ylabel('Rainfall (mm)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Decadal seasonal comparison
        self.data['Decade'] = (self.data['Year'] // 10) * 10
        decadal_seasonal = self.data.groupby(['Decade', 'Season'])['Temperature_C'].mean().unstack()
        decadal_seasonal[seasons_order].plot(ax=axes[1, 1], marker='o')
        axes[1, 1].set_title('Seasonal Temperature by Decade', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('Decade')
        axes[1, 1].set_ylabel('Temperature (°C)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved static image: {save_path}")
        return save_path
    
    def plot_correlation_heatmap(self, save_path='outputs/figures/correlation_heatmap.png'):
        """
        Create correlation heatmap for climate variables
        Saves as static PNG file
        """
        numeric_cols = ['Temperature_C', 'Rainfall_mm', 'Humidity_Percent', 'CO2_ppm', 'Year']
        corr_data = self.data[numeric_cols].dropna()
        correlation_matrix = corr_data.corr()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=1, cbar_kws={"shrink": 0.8},
                   fmt='.2f', ax=ax)
        
        ax.set_title('Climate Variables Correlation Matrix', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved static image: {save_path}")
        return save_path
    
    def plot_decadal_comparison(self, save_path='outputs/figures/decadal_comparison.png'):
        """
        Compare climate variables across decades
        Saves as static PNG file
        """
        self.data['Decade'] = (self.data['Year'] // 10) * 10
        
        decadal_temp = self.data.groupby('Decade')['Temperature_C'].mean()
        decadal_rainfall = self.data.groupby('Decade')['Rainfall_mm'].mean()
        decadal_co2 = self.data.groupby('Decade')['CO2_ppm'].mean()
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].bar(decadal_temp.index, decadal_temp.values, color='coral')
        axes[0].set_title('Average Temperature by Decade', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Decade')
        axes[0].set_ylabel('Temperature (°C)')
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].grid(True, alpha=0.3)
        
        axes[1].bar(decadal_rainfall.index, decadal_rainfall.values, color='skyblue')
        axes[1].set_title('Average Rainfall by Decade', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Decade')
        axes[1].set_ylabel('Rainfall (mm)')
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].grid(True, alpha=0.3)
        
        axes[2].bar(decadal_co2.index, decadal_co2.values, color='lightgreen')
        axes[2].set_title('Average CO2 by Decade', fontsize=12, fontweight='bold')
        axes[2].set_xlabel('Decade')
        axes[2].set_ylabel('CO2 (ppm)')
        axes[2].tick_params(axis='x', rotation=45)
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved static image: {save_path}")
        return save_path
    
    def create_all_static_visualizations(self):
        """
        Generate all static visualizations at once
        """
        print("\n📊 Generating static visualizations...")
        print("-" * 40)
        
        images = []
        images.append(self.plot_temperature_trend())
        images.append(self.plot_seasonal_patterns())
        images.append(self.plot_correlation_heatmap())
        images.append(self.plot_decadal_comparison())
        
        print("-" * 40)
        print(f"✅ Generated {len(images)} static images in outputs/figures/")
        
        return images

# Test the module
if __name__ == "__main__":
    from data_loader import ClimateDataLoader
    
    # Load data
    loader = ClimateDataLoader()
    df = loader.generate_synthetic_climate_data()
    
    # Create static visualizations
    viz = ClimateVisualizer(df)
    viz.create_all_static_visualizations()