"""
Climate Trend Analyzer - Main Execution Script
Run this to execute the complete climate analysis pipeline and generate static images
"""
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data_loader import ClimateDataLoader
from src.preprocessor import ClimatePreprocessor
from src.trend_analyzer import ClimateTrendAnalyzer
from src.anomaly_detector import ClimateAnomalyDetector
from src.visualizer import ClimateVisualizer

def ensure_directories():
    """Create necessary directories if they don't exist"""
    directories = [
        'data/raw',
        'data/processed',
        'outputs/figures',
        'outputs/reports',
        'images',
        'app'
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    print("✅ Directories created/verified")

def generate_climate_data():
    """Generate or load climate data"""
    print("\n" + "="*60)
    print("STEP 1: DATA COLLECTION")
    print("="*60)
    
    loader = ClimateDataLoader()
    
    # Generate synthetic climate data (1900-2024)
    df = loader.generate_synthetic_climate_data(start_year=1900, end_year=2024)
    
    # Add artificial anomalies for demonstration
    df = loader.add_anomalies(anomaly_percentage=2)
    
    # Save raw data
    raw_path = 'data/raw/climate_data.csv'
    df.to_csv(raw_path, index=False)
    print(f"✅ Raw data saved to {raw_path}")
    
    # Display summary
    summary = loader.get_summary()
    print(f"\n📊 Dataset Summary:")
    print(f"  - Shape: {summary['shape']}")
    print(f"  - Date Range: {summary['date_range']}")
    print(f"  - Columns: {summary['columns']}")
    
    return df

def preprocess_data(df):
    """Clean and preprocess the climate data"""
    print("\n" + "="*60)
    print("STEP 2: DATA PREPROCESSING")
    print("="*60)
    
    preprocessor = ClimatePreprocessor()
    preprocessor.load_data(df)
    
    # Handle missing values
    preprocessor.handle_missing_values(strategy='ffill')
    
    # Remove outliers
    preprocessor.remove_outliers('Temperature_C', method='iqr')
    
    # Create additional features
    preprocessor.create_features()
    
    # Normalize data
    preprocessor.normalize_data(columns=['Temperature_C', 'Rainfall_mm'], method='standard')
    
    # Get processed data
    processed_df = preprocessor.get_processed_data()
    
    # Save processed data
    processed_path = 'data/processed/cleaned_climate_data.csv'
    preprocessor.save_processed_data(processed_path)
    
    # Display preprocessing report
    report = preprocessor.get_preprocessing_report()
    print(f"\n📊 Preprocessing Report:")
    print(f"  - Processed Shape: {report['processed_shape']}")
    print(f"  - Missing Values Remaining: {report['missing_values_remaining']}")
    print(f"  - Memory Usage: {report['memory_usage']:.2f} MB")
    
    return processed_df

def analyze_trends(processed_df):
    """Perform climate trend analysis"""
    print("\n" + "="*60)
    print("STEP 3: TREND ANALYSIS")
    print("="*60)
    
    analyzer = ClimateTrendAnalyzer(processed_df)
    
    # Get comprehensive trend report
    trend_report = analyzer.get_comprehensive_report()
    
    # Display key findings
    print("\n📈 Key Trend Findings:")
    
    if 'temperature_analysis' in trend_report and 'trend' in trend_report['temperature_analysis']:
        temp_trend = trend_report['temperature_analysis']['trend']
        print(f"\n  🌡️ Temperature:")
        print(f"    - Trend Direction: {temp_trend.get('trend_direction', 'N/A')}")
        print(f"    - Annual Rate: {temp_trend.get('annual_sen_slope', temp_trend.get('slope_per_year', 0)):.4f}°C/year")
    
    if 'warming_rate_analysis' in trend_report:
        warming = trend_report['warming_rate_analysis']
        print(f"\n  📊 Warming Rate:")
        if 'overall' in warming:
            print(f"    - Overall Rate: {warming['overall']['annual_rate_c_per_year']:.4f}°C/year")
        if 'post_1980' in warming:
            print(f"    - Post-1980 Rate: {warming['post_1980']['annual_rate_c_per_year']:.4f}°C/year")
    
    # Save report
    report_path = 'outputs/reports/climate_analysis_report.md'
    with open(report_path, 'w') as f:
        f.write("# Climate Trend Analysis Report\n\n")
        f.write(f"## Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## Key Findings\n")
        if 'temperature_analysis' in trend_report:
            f.write(f"- Temperature Trend: {trend_report['temperature_analysis'].get('trend', {})}\n")
    
    print(f"\n✅ Detailed report saved to {report_path}")
    
    return analyzer, trend_report

def detect_anomalies(processed_df):
    """Detect anomalies in climate data"""
    print("\n" + "="*60)
    print("STEP 4: ANOMALY DETECTION")
    print("="*60)
    
    detector = ClimateAnomalyDetector(processed_df)
    
    # Statistical anomaly detection
    temp_anomalies = detector.statistical_anomaly_detection('Temperature_C', method='zscore', threshold=2.5)
    rainfall_anomalies = detector.statistical_anomaly_detection('Rainfall_mm', method='zscore', threshold=2.5)
    
    # Isolation Forest for multivariate anomalies
    forest_anomalies = detector.isolation_forest_anomaly_detection(contamination=0.05)
    
    # Get summary
    anomaly_summary = detector.get_anomaly_summary()
    
    # Display findings
    print(f"\n⚠️ Anomaly Detection Results:")
    print(f"  - Temperature Anomalies: {temp_anomalies['anomaly_count']} ({temp_anomalies['anomaly_percentage']:.2f}%)")
    print(f"  - Rainfall Anomalies: {rainfall_anomalies['anomaly_count']} ({rainfall_anomalies['anomaly_percentage']:.2f}%)")
    print(f"  - Multivariate Anomalies: {forest_anomalies['anomaly_count']} ({forest_anomalies['anomaly_percentage']:.2f}%)")
    
    return detector, anomaly_summary

def create_static_visualizations(processed_df):
    """Create and save static visualizations"""
    print("\n" + "="*60)
    print("STEP 5: STATIC VISUALIZATION (SAVING IMAGES)")
    print("="*60)
    
    viz = ClimateVisualizer(processed_df)
    
    # Create all static visualizations
    images = viz.create_all_static_visualizations()
    
    print(f"\n✅ Saved {len(images)} static images to 'outputs/figures/'")
    print("   These images can be used in README.md and reports")
    
    return viz

def generate_summary_report(trend_report, anomaly_summary, processed_df):
    """Generate final summary report"""
    print("\n" + "="*60)
    print("STEP 6: FINAL SUMMARY")
    print("="*60)
    
    summary = {
        'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'data_range': f"{processed_df['Date'].min()} to {processed_df['Date'].max()}",
        'total_records': len(processed_df),
        'key_findings': {}
    }
    
    # Extract key findings
    if 'temperature_analysis' in trend_report and 'trend' in trend_report['temperature_analysis']:
        temp_trend = trend_report['temperature_analysis']['trend']
        summary['key_findings']['temperature'] = {
            'trend': temp_trend.get('trend_direction', 'N/A'),
            'rate': temp_trend.get('annual_sen_slope', temp_trend.get('slope_per_year', 0))
        }
    
    summary['anomaly_stats'] = {
        'total_anomalies': sum(anomaly_summary.get('total_anomalies_by_method', {}).values())
    }
    
    # Print summary
    print("\n" + "="*60)
    print("📊 CLIMATE TREND ANALYZER - FINAL SUMMARY")
    print("="*60)
    print(f"📅 Analysis Date: {summary['analysis_date']}")
    print(f"📆 Data Period: {summary['data_range']}")
    print(f"📝 Total Records Analyzed: {summary['total_records']}")
    
    print("\n📈 Key Findings:")
    for key, value in summary['key_findings'].items():
        print(f"  - {key.upper()}: {value['trend']} at {value['rate']:.4f} units/year")
    
    print(f"\n⚠️ Anomalies Detected: {summary['anomaly_stats']['total_anomalies']}")
    
    # Save summary
    summary_path = 'outputs/reports/final_summary.txt'
    with open(summary_path, 'w') as f:
        f.write("CLIMATE TREND ANALYZER - FINAL SUMMARY\n")
        f.write("="*50 + "\n\n")
        for key, value in summary.items():
            f.write(f"{key}: {value}\n")
    
    print(f"\n✅ Final summary saved to {summary_path}")
    
    return summary

def main():
    """Main execution function"""
    print("\n" + "="*60)
    print("🌍 CLIMATE TREND ANALYZER")
    print("Data Science Project for Climate Analysis")
    print("="*60)
    
    # Create directories
    ensure_directories()
    
    try:
        # Step 1: Generate/Load data
        raw_data = generate_climate_data()
        
        # Step 2: Preprocess data
        processed_data = preprocess_data(raw_data)
        
        # Step 3: Analyze trends
        analyzer, trend_report = analyze_trends(processed_data)
        
        # Step 4: Detect anomalies
        detector, anomaly_summary = detect_anomalies(processed_data)
        
        # Step 5: Create static visualizations (SAVES IMAGES)
        viz = create_static_visualizations(processed_data)
        
        # Step 6: Generate summary
        summary = generate_summary_report(trend_report, anomaly_summary, processed_data)
        
        print("\n" + "="*60)
        print("✅ PROJECT COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\n📁 Output files created:")
        print("  📊 data/raw/climate_data.csv")
        print("  📊 data/processed/cleaned_climate_data.csv")
        print("  🖼️ outputs/figures/*.png (static images for documentation)")
        print("  📄 outputs/reports/climate_analysis_report.md")
        print("  📄 outputs/reports/final_summary.txt")
        
        print("\n" + "="*60)
        print("🚀 TO START THE INTERACTIVE DASHBOARD:")
        print("="*60)
        print("\n  Option 1 (Recommended):")
        print("    python run_dashboard.py")
        print("\n  Option 2:")
        print("    streamlit run app/streamlit_app.py")
        print("\n" + "="*60)
        
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        print("Please check your setup and try again.")
        raise

if __name__ == "__main__":
    main()