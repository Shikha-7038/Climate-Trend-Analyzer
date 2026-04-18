"""
Trend Analyzer Module - Detects and analyzes climate trends
"""
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.seasonal import seasonal_decompose
import warnings
warnings.filterwarnings('ignore')

class ClimateTrendAnalyzer:
    """
    Analyzes long-term trends in climate data
    """
    
    def __init__(self, data):
        """
        Initialize with processed climate data
        """
        self.data = data.copy()
        self.trend_results = {}
        
        # Ensure Date is index for time series operations
        if 'Date' in self.data.columns:
            self.data['Date'] = pd.to_datetime(self.data['Date'])
            self.data.set_index('Date', inplace=True)
    
    def calculate_trend(self, column, method='linear'):
        """
        Calculate trend for a specific column
        
        Parameters:
        - column: Column name to analyze
        - method: 'linear' or 'mann_kendall'
        
        Returns:
        - Dictionary with trend statistics
        """
        # Remove NaN values
        series = self.data[column].dropna()
        
        if len(series) < 2:
            return {'error': 'Insufficient data for trend analysis'}
        
        if method == 'linear':
            # Linear regression trend
            x = np.arange(len(series)).reshape(-1, 1)
            y = series.values.reshape(-1, 1)
            
            model = LinearRegression()
            model.fit(x, y)
            
            trend_slope = float(model.coef_[0][0])  # Change per time step
            trend_intercept = float(model.intercept_[0])
            r_squared = float(model.score(x, y))
            
            # Convert slope to annual change
            if len(series) > 12:  # Monthly data
                annual_slope = trend_slope * 12
            else:
                annual_slope = trend_slope
            
            result = {
                'method': 'linear_regression',
                'slope_per_period': trend_slope,
                'slope_per_year': annual_slope,
                'intercept': trend_intercept,
                'r_squared': r_squared,
                'trend_direction': 'increasing' if trend_slope > 0 else 'decreasing',
                'total_change': float(trend_slope * len(series)) if len(series) > 0 else 0
            }
        
        elif method == 'mann_kendall':
            # Mann-Kendall trend test (non-parametric)
            n = len(series)
            s = 0
            
            for i in range(n-1):
                for j in range(i+1, n):
                    s += np.sign(series.iloc[j] - series.iloc[i])
            
            # Variance calculation
            unique_values = series.value_counts()
            tie_correction = sum(v * (v-1) * (2*v + 5) for v in unique_values)
            var_s = (n * (n-1) * (2*n + 5) - tie_correction) / 18
            
            if s > 0:
                z = (s - 1) / np.sqrt(var_s)
            elif s < 0:
                z = (s + 1) / np.sqrt(var_s)
            else:
                z = 0
            
            # P-value (two-tailed)
            p_value = 2 * (1 - stats.norm.cdf(abs(z)))
            
            # Sen's slope estimator
            slopes = []
            for i in range(n-1):
                for j in range(i+1, n):
                    slope = (series.iloc[j] - series.iloc[i]) / (j - i)
                    slopes.append(slope)
            
            sen_slope = np.median(slopes) if slopes else 0
            
            result = {
                'method': 'mann_kendall',
                'kendall_s': float(s),
                'z_score': float(z),
                'p_value': float(p_value),
                'sen_slope': float(sen_slope),
                'annual_sen_slope': float(sen_slope * 12) if len(series) > 12 else float(sen_slope),
                'trend_direction': 'increasing' if sen_slope > 0 else 'decreasing' if sen_slope < 0 else 'no trend',
                'statistically_significant': p_value < 0.05
            }
        
        self.trend_results[column] = result
        return result
    
    def get_decadal_trends(self, column):
        """
        Calculate trends by decade
        """
        self.data['Decade'] = (self.data.index.year // 10) * 10
        decadal_means = self.data.groupby('Decade')[column].mean()
        
        trends = {}
        decades = sorted(decadal_means.index)
        
        for i in range(len(decades)-1):
            decade_start = decades[i]
            decade_end = decades[i+1]
            change = decadal_means[decade_end] - decadal_means[decade_start]
            percent_change = (change / decadal_means[decade_start]) * 100 if decadal_means[decade_start] != 0 else 0
            
            trends[f"{decade_start}-{decade_end}"] = {
                'start_value': float(decadal_means[decade_start]),
                'end_value': float(decadal_means[decade_end]),
                'absolute_change': float(change),
                'percent_change': float(percent_change)
            }
        
        return trends
    
    def seasonal_trend_analysis(self, column):
        """
        Analyze seasonal patterns and trends
        """
        # Add season if not present
        if 'Season' not in self.data.columns:
            self.data['Month'] = self.data.index.month
            self.data['Season'] = self.data['Month'].map({
                12: 'Winter', 1: 'Winter', 2: 'Winter',
                3: 'Spring', 4: 'Spring', 5: 'Spring',
                6: 'Summer', 7: 'Summer', 8: 'Summer',
                9: 'Fall', 10: 'Fall', 11: 'Fall'
            })
        
        # Calculate seasonal means
        seasonal_means = self.data.groupby('Season')[column].mean()
        
        # Calculate seasonal trends over time
        seasonal_trends = {}
        for season in ['Winter', 'Spring', 'Summer', 'Fall']:
            season_data = self.data[self.data['Season'] == season]
            if len(season_data) > 0:
                # Linear trend for this season
                x = np.arange(len(season_data)).reshape(-1, 1)
                y = season_data[column].values.reshape(-1, 1)
                model = LinearRegression()
                model.fit(x, y)
                
                seasonal_trends[season] = {
                    'mean_value': float(season_data[column].mean()),
                    'trend_slope': float(model.coef_[0][0]),
                    'trend_per_year': float(model.coef_[0][0] * 12),
                    'max_value': float(season_data[column].max()),
                    'min_value': float(season_data[column].min())
                }
        
        return {
            'seasonal_averages': seasonal_means.to_dict(),
            'seasonal_trends': seasonal_trends
        }
    
    def warming_rate_analysis(self):
        """
        Calculate warming rates for different time periods
        """
        if 'Temperature_C' not in self.data.columns:
            return "Temperature column not found"
        
        warming_rates = {}
        
        # Calculate warming rate for entire period
        overall = self.calculate_trend('Temperature_C')
        warming_rates['overall'] = {
            'annual_rate_c_per_year': overall.get('annual_sen_slope', overall.get('slope_per_year', 0)),
            'total_warming_c': overall.get('total_change', 0)
        }
        
        # Split into periods (pre-1980, post-1980)
        pre_1980_data = self.data[self.data.index.year < 1980]['Temperature_C']
        post_1980_data = self.data[self.data.index.year >= 1980]['Temperature_C']
        
        if len(pre_1980_data) > 1:
            temp_data = pd.DataFrame({'value': pre_1980_data})
            x = np.arange(len(temp_data)).reshape(-1, 1)
            y = temp_data['value'].values.reshape(-1, 1)
            model = LinearRegression()
            model.fit(x, y)
            warming_rates['pre_1980'] = {
                'annual_rate_c_per_year': float(model.coef_[0][0] * 12),
                'period': '1900-1980'
            }
        
        if len(post_1980_data) > 1:
            temp_data = pd.DataFrame({'value': post_1980_data})
            x = np.arange(len(temp_data)).reshape(-1, 1)
            y = temp_data['value'].values.reshape(-1, 1)
            model = LinearRegression()
            model.fit(x, y)
            warming_rates['post_1980'] = {
                'annual_rate_c_per_year': float(model.coef_[0][0] * 12),
                'period': '1980-2024'
            }
            
            # Calculate acceleration only if both periods exist
            if 'pre_1980' in warming_rates:
                warming_rates['acceleration'] = (
                    warming_rates['post_1980']['annual_rate_c_per_year'] - 
                    warming_rates['pre_1980']['annual_rate_c_per_year']
                )
            else:
                warming_rates['acceleration'] = None
        
        return warming_rates
    
    def get_comprehensive_report(self):
        """
        Generate a comprehensive trend analysis report
        """
        report = {
            'temperature_analysis': {},
            'rainfall_analysis': {},
            'humidity_analysis': {},
            'co2_analysis': {}
        }
        
        # Analyze each climate variable
        variables = {
            'Temperature_C': 'temperature_analysis',
            'Rainfall_mm': 'rainfall_analysis', 
            'Humidity_Percent': 'humidity_analysis',
            'CO2_ppm': 'co2_analysis'
        }
        
        for var, key in variables.items():
            if var in self.data.columns:
                try:
                    trend = self.calculate_trend(var)
                    decadal = self.get_decadal_trends(var)
                    seasonal = self.seasonal_trend_analysis(var)
                    
                    report[key] = {
                        'trend': trend,
                        'decadal_trends': decadal,
                        'seasonal_patterns': seasonal
                    }
                except Exception as e:
                    report[key] = {'error': str(e)}
        
        # Add warming rate analysis
        warming_rates = self.warming_rate_analysis()
        if isinstance(warming_rates, dict):
            report['warming_rate_analysis'] = warming_rates
        else:
            report['warming_rate_analysis'] = {'error': warming_rates}
        
        return report

# Test the module
if __name__ == "__main__":
    from data_loader import ClimateDataLoader
    from preprocessor import ClimatePreprocessor
    
    # Load and preprocess data
    loader = ClimateDataLoader()
    df = loader.generate_synthetic_climate_data()
    
    preprocessor = ClimatePreprocessor()
    preprocessor.load_data(df)
    preprocessor.create_features()
    processed_df = preprocessor.get_processed_data()
    
    # Analyze trends
    analyzer = ClimateTrendAnalyzer(processed_df)
    report = analyzer.get_comprehensive_report()
    
    print("\n=== TREND ANALYSIS REPORT ===")
    print(f"Temperature Trend: {report['temperature_analysis'].get('trend', 'N/A')}")
    print(f"\nWarming Rate Analysis: {report.get('warming_rate_analysis', 'N/A')}")