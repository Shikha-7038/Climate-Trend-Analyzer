"""
Run Dashboard - ONE COMMAND to start the interactive dashboard
This script checks if data exists and launches the Streamlit app
"""

import subprocess
import sys
import os

def check_data():
    """Check if climate data exists, if not, run main.py"""
    if not os.path.exists('data/processed/cleaned_climate_data.csv'):
        print("⚠️ No processed data found. Running main.py to generate data...")
        print("-" * 50)
        
        # Run main.py to generate data and static images
        result = subprocess.run([sys.executable, 'main.py'], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Data generated successfully!")
        else:
            print("❌ Error generating data:")
            print(result.stderr)
            return False
    
    return True

def launch_dashboard():
    """Launch the Streamlit dashboard"""
    print("\n" + "="*60)
    print("🚀 CLIMATE TREND ANALYZER - DASHBOARD")
    print("="*60)
    print("\n📊 Starting interactive dashboard...")
    print("🌐 Dashboard will open in your browser")
    print("⏹️ Press Ctrl+C to stop the dashboard\n")
    print("="*60 + "\n")
    
    # Launch Streamlit app
    subprocess.run([
        sys.executable, '-m', 'streamlit', 'run', 
        'app/streamlit_app.py',
        '--server.port', '8501',
        '--server.address', 'localhost'
    ])

if __name__ == "__main__":
    # Check and generate data if needed
    if check_data():
        # Launch the dashboard
        launch_dashboard()
    else:
        print("\n❌ Failed to prepare data. Please run 'python main.py' manually first.")