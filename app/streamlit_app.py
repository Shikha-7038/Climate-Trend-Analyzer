"""
Climate Trend Analyzer - Interactive Streamlit Dashboard
Run with: streamlit run app/streamlit_app.py
OR: python run_dashboard.py (even easier!)
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
from datetime import datetime

# ==================== PAGE CONFIGURATION ====================
st.set_page_config(
    page_title="Climate Trend Analyzer",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CUSTOM CSS FOR BETTER LOOKS ====================
# ==================== CUSTOM CSS FOR BETTER LOOKS ====================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
    
    /* INSIGHT BOX - Darker green background with white text */
    .insight-box {
        background-color: #2E7D32;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #81C784;
        margin: 1rem 0;
        color: #FFFFFF !important;
    }
    .insight-box strong {
        color: #FFFFFF !important;
        font-weight: bold;
    }
    .insight-box br {
        color: #FFFFFF;
    }
    
    /* WARNING BOX - Darker orange/amber background with white text */
    .warning-box {
        background-color: #E65100;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #FFB74D;
        margin: 1rem 0;
        color: #FFFFFF !important;
    }
    .warning-box strong {
        color: #FFFFFF !important;
        font-weight: bold;
    }
    .warning-box br {
        color: #FFFFFF;
    }
    
    /* Additional style for any text inside these boxes */
    .insight-box span, .insight-box div, .insight-box p {
        color: #FFFFFF !important;
    }
    .warning-box span, .warning-box div, .warning-box p {
        color: #FFFFFF !important;
    }
</style>
""", unsafe_allow_html=True)

# ==================== LOAD DATA FUNCTION ====================
@st.cache_data
def load_data():
    """Load processed climate data"""
    # Try to load from processed folder first
    processed_path = 'data/processed/cleaned_climate_data.csv'
    raw_path = 'data/raw/climate_data.csv'
    
    if os.path.exists(processed_path):
        df = pd.read_csv(processed_path)
        st.success("✅ Loaded processed climate data")
    elif os.path.exists(raw_path):
        df = pd.read_csv(raw_path)
        st.info("📊 Loaded raw climate data (run main.py for full processing)")
    else:
        st.error("❌ No data found! Please run 'python main.py' first to generate data.")
        st.stop()
    
    df['Date'] = pd.to_datetime(df['Date'])
    return df

# ==================== SIDEBAR FILTERS ====================
def create_sidebar_filters(df):
    """Create interactive filters in sidebar"""
    st.sidebar.markdown("## 🎛️ Dashboard Controls")
    
    # Date range selector
    min_year = int(df['Year'].min())
    max_year = int(df['Year'].max())
    
    year_range = st.sidebar.slider(
        "📅 Select Year Range",
        min_value=min_year,
        max_value=max_year,
        value=(min_year, max_year),
        step=5
    )
    
    # Variable selector for multi-line chart
    st.sidebar.markdown("### 📊 Variables to Display")
    all_vars = {
        'Temperature_C': '🌡️ Temperature (°C)',
        'Rainfall_mm': '🌧️ Rainfall (mm)',
        'Humidity_Percent': '💧 Humidity (%)',
        'CO2_ppm': '🏭 CO2 (ppm)'
    }
    
    selected_vars = []
    for var, label in all_vars.items():
        if var in df.columns:
            if st.sidebar.checkbox(label, value=(var == 'Temperature_C')):
                selected_vars.append(var)
    
    # Anomaly detection threshold
    st.sidebar.markdown("### ⚠️ Anomaly Detection")
    anomaly_threshold = st.sidebar.slider(
        "Anomaly Threshold (Z-score)",
        min_value=1.0,
        max_value=3.5,
        value=2.5,
        step=0.1,
        help="Higher values = fewer anomalies detected"
    )
    
    # Rolling average window
    st.sidebar.markdown("### 📈 Smoothing")
    rolling_window = st.sidebar.selectbox(
        "Rolling Average Window",
        options=['None', '1 Year', '5 Years', '10 Years'],
        index=2
    )
    
    rolling_map = {
        'None': 1,
        '1 Year': 12,
        '5 Years': 60,
        '10 Years': 120
    }
    rolling_value = rolling_map[rolling_window]
    
    return year_range, selected_vars, anomaly_threshold, rolling_value

# ==================== MAIN DASHBOARD ====================
def main():
    # Header
    st.markdown('<div class="main-header">🌍 Climate Trend Analyzer</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Interactive Dashboard | 1900-2024 Climate Data Analysis</div>', unsafe_allow_html=True)
    
    # Load data
    with st.spinner("Loading climate data..."):
        df = load_data()
    
    # Create filters
    year_range, selected_vars, anomaly_threshold, rolling_window = create_sidebar_filters(df)
    
    # Filter data based on selection
    filtered_df = df[(df['Year'] >= year_range[0]) & (df['Year'] <= year_range[1])].copy()
    
    # Apply rolling average if selected
    if rolling_window > 1:
        for var in selected_vars:
            if var in filtered_df.columns:
                filtered_df[f'{var}_smoothed'] = filtered_df[var].rolling(window=rolling_window, min_periods=1).mean()
    
    # ==================== KEY METRICS ROW ====================
    st.markdown("## 📈 Key Climate Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        temp_change = filtered_df['Temperature_C'].iloc[-1] - filtered_df['Temperature_C'].iloc[0]
        st.metric(
            label="🌡️ Temperature Change",
            value=f"{temp_change:+.2f}°C",
            delta=f"{temp_change:+.2f}°C over period",
            delta_color="inverse"
        )
    
    with col2:
        avg_temp = filtered_df['Temperature_C'].mean()
        st.metric(label="📊 Average Temperature", value=f"{avg_temp:.1f}°C")
    
    with col3:
        avg_rain = filtered_df['Rainfall_mm'].mean()
        st.metric(label="🌧️ Average Rainfall", value=f"{avg_rain:.1f} mm")
    
    with col4:
        max_temp = filtered_df['Temperature_C'].max()
        max_temp_year = filtered_df[filtered_df['Temperature_C'] == max_temp]['Year'].iloc[0]
        st.metric(label="🔥 Record High", value=f"{max_temp:.1f}°C", delta=f"in {int(max_temp_year)}")
    
    # ==================== MAIN CHARTS ====================
    
    # Row 1: Temperature Trend (Main Chart)
    st.markdown("## 🌡️ Temperature Trend Analysis")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig_temp = go.Figure()
        
        # Add temperature line
        y_col = 'Temperature_C_smoothed' if rolling_window > 1 and 'Temperature_C_smoothed' in filtered_df.columns else 'Temperature_C'
        fig_temp.add_trace(go.Scatter(
            x=filtered_df['Date'],
            y=filtered_df[y_col],
            mode='lines',
            name='Temperature',
            line=dict(color='red', width=2),
            fill='tozeroy',
            fillcolor='rgba(255,0,0,0.1)'
        ))
        
        # Add rolling average if different
        if rolling_window > 1 and 'Temperature_C_smoothed' in filtered_df.columns:
            fig_temp.add_trace(go.Scatter(
                x=filtered_df['Date'],
                y=filtered_df['Temperature_C'],
                mode='lines',
                name='Monthly Data',
                line=dict(color='lightcoral', width=0.5, dash='dot'),
                opacity=0.5
            ))
        
        fig_temp.update_layout(
            title=f"Temperature Trend ({year_range[0]}-{year_range[1]})",
            xaxis_title="Year",
            yaxis_title="Temperature (°C)",
            height=450,
            hovermode='x unified',
            template='plotly_white'
        )
        
        st.plotly_chart(fig_temp, use_container_width=True)
    
    with col2:
        # Temperature anomaly gauge
        baseline = df[df['Year'].between(1951, 1980)]['Temperature_C'].mean()
        current_temp = filtered_df['Temperature_C'].iloc[-1]
        anomaly = current_temp - baseline
        
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = current_temp,
            delta = {'reference': baseline, 'relative': True},
            title = {'text': f"Current Temp vs 1951-80 Baseline<br>(+{anomaly:.2f}°C)"},
            gauge = {
                'axis': {'range': [None, baseline + 3]},
                'bar': {'color': "red" if anomaly > 0 else "blue"},
                'steps': [
                    {'range': [baseline - 1, baseline], 'color': "lightblue"},
                    {'range': [baseline, baseline + 1], 'color': "lightcoral"},
                    {'range': [baseline + 1, baseline + 3], 'color': "salmon"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': baseline
                }
            }
        ))
        fig_gauge.update_layout(height=300)
        st.plotly_chart(fig_gauge, use_container_width=True)
    
    # ==================== MULTI-VARIABLE COMPARISON ====================
    if len(selected_vars) > 0:
        st.markdown("## 📊 Multi-Variable Comparison")
        
        fig_multi = go.Figure()
        
        colors = {'Temperature_C': 'red', 'Rainfall_mm': 'blue', 'Humidity_Percent': 'green', 'CO2_ppm': 'orange'}
        
        for var in selected_vars:
            y_col = f'{var}_smoothed' if rolling_window > 1 and f'{var}_smoothed' in filtered_df.columns else var
            fig_multi.add_trace(go.Scatter(
                x=filtered_df['Date'],
                y=filtered_df[y_col],
                mode='lines',
                name=var.replace('_', ' '),
                line=dict(color=colors.get(var, 'gray'), width=2)
            ))
        
        fig_multi.update_layout(
            title="Climate Variables Over Time",
            xaxis_title="Year",
            yaxis_title="Value",
            height=500,
            hovermode='x unified',
            template='plotly_white'
        )
        
        st.plotly_chart(fig_multi, use_container_width=True)
    
    # ==================== SEASONAL PATTERNS ====================
    st.markdown("## 🌸 Seasonal Pattern Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Monthly average temperatures
        monthly_avg = filtered_df.groupby('Month')['Temperature_C'].mean()
        
        fig_seasonal = go.Figure()
        fig_seasonal.add_trace(go.Scatter(
            x=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
            y=monthly_avg.values,
            mode='lines+markers',
            name='Average Temperature',
            line=dict(color='red', width=3),
            marker=dict(size=10)
        ))
        
        fig_seasonal.update_layout(
            title="Monthly Average Temperature",
            xaxis_title="Month",
            yaxis_title="Temperature (°C)",
            height=400,
            template='plotly_white'
        )
        
        st.plotly_chart(fig_seasonal, use_container_width=True)
    
    with col2:
        # Seasonal rainfall
        filtered_df['Season'] = filtered_df['Month'].map({
            12: 'Winter', 1: 'Winter', 2: 'Winter',
            3: 'Spring', 4: 'Spring', 5: 'Spring',
            6: 'Summer', 7: 'Summer', 8: 'Summer',
            9: 'Fall', 10: 'Fall', 11: 'Fall'
        })
        
        seasonal_rain = filtered_df.groupby('Season')['Rainfall_mm'].mean()
        season_order = ['Winter', 'Spring', 'Summer', 'Fall']
        seasonal_rain = seasonal_rain.reindex(season_order)
        
        fig_rain = go.Figure()
        fig_rain.add_trace(go.Bar(
            x=seasonal_rain.index,
            y=seasonal_rain.values,
            marker_color=['lightblue', 'lightgreen', 'gold', 'orange'],
            text=seasonal_rain.values.round(1),
            textposition='auto'
        ))
        
        fig_rain.update_layout(
            title="Average Rainfall by Season",
            xaxis_title="Season",
            yaxis_title="Rainfall (mm)",
            height=400,
            template='plotly_white'
        )
        
        st.plotly_chart(fig_rain, use_container_width=True)
    
    # ==================== ANOMALY DETECTION ====================
    st.markdown("## ⚠️ Climate Anomaly Detection")
    
    # Detect anomalies using Z-score
    mean_temp = filtered_df['Temperature_C'].mean()
    std_temp = filtered_df['Temperature_C'].std()
    filtered_df['Z_Score'] = (filtered_df['Temperature_C'] - mean_temp) / std_temp
    anomalies = filtered_df[abs(filtered_df['Z_Score']) > anomaly_threshold]
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.metric(
            label="📊 Anomalies Detected",
            value=len(anomalies),
            delta=f"{len(anomalies)/len(filtered_df)*100:.1f}% of data",
            delta_color="inverse"
        )
        
        if len(anomalies) > 0:
            st.markdown(f"""
            <div class="warning-box">
                <strong>⚠️ Recent Anomalies:</strong><br>
                • Most recent anomaly: {anomalies['Date'].iloc[-1].strftime('%Y-%m')}<br>
                • Highest anomaly: {anomalies['Temperature_C'].max():.1f}°C<br>
                • Anomaly threshold: ±{anomaly_threshold} σ
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        # Anomaly plot
        fig_anomaly = go.Figure()
        
        # Normal points
        normal = filtered_df[abs(filtered_df['Z_Score']) <= anomaly_threshold]
        fig_anomaly.add_trace(go.Scatter(
            x=normal['Date'],
            y=normal['Temperature_C'],
            mode='markers',
            name='Normal',
            marker=dict(color='blue', size=4, opacity=0.5)
        ))
        
        # Anomaly points
        fig_anomaly.add_trace(go.Scatter(
            x=anomalies['Date'],
            y=anomalies['Temperature_C'],
            mode='markers',
            name='Anomaly',
            marker=dict(color='red', size=10, symbol='x')
        ))
        
        fig_anomaly.update_layout(
            title=f"Temperature Anomalies (|Z| > {anomaly_threshold})",
            xaxis_title="Year",
            yaxis_title="Temperature (°C)",
            height=400,
            template='plotly_white'
        )
        
        st.plotly_chart(fig_anomaly, use_container_width=True)
    
    # ==================== DECADAL COMPARISON ====================
    st.markdown("## 📅 Decadal Climate Comparison")
    
    filtered_df['Decade'] = (filtered_df['Year'] // 10) * 10
    decadal_stats = filtered_df.groupby('Decade').agg({
        'Temperature_C': ['mean', 'min', 'max'],
        'Rainfall_mm': 'mean',
        'CO2_ppm': 'mean'
    }).round(2)
    
    # Show decadal table
    st.dataframe(decadal_stats, use_container_width=True)
    
    # Decadal bar chart
    fig_decadal = go.Figure()
    fig_decadal.add_trace(go.Bar(
        x=decadal_stats.index,
        y=decadal_stats['Temperature_C']['mean'],
        name='Average Temperature',
        marker_color='coral'
    ))
    
    fig_decadal.update_layout(
        title="Average Temperature by Decade",
        xaxis_title="Decade",
        yaxis_title="Temperature (°C)",
        height=400,
        template='plotly_white'
    )
    
    st.plotly_chart(fig_decadal, use_container_width=True)
    
    # ==================== INSIGHTS & RECOMMENDATIONS ====================
    st.markdown("## 💡 Key Insights & Recommendations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="insight-box">
            <strong>📈 Temperature Trend:</strong><br>
            Global temperatures have increased significantly over the analysis period.
            The rate of warming has accelerated in recent decades, consistent with 
            global climate change observations.
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="insight-box">
            <strong>🌧️ Rainfall Patterns:</strong><br>
            Rainfall shows increased variability with more extreme events.
            This pattern suggests changing weather patterns that could affect
            agriculture and water resource management.
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="insight-box">
            <strong>⚠️ Anomaly Frequency:</strong><br>
            Extreme temperature events have become more frequent, especially
            in the last 30 years. This trend indicates increasing climate volatility
            and the need for adaptive strategies.
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="insight-box">
            <strong>🎯 Recommendations:</strong><br>
            • Monitor seasonal patterns for agricultural planning<br>
            • Prepare for more frequent extreme weather events<br>
            • Consider climate trends in long-term infrastructure planning
        </div>
        """, unsafe_allow_html=True)
    
    # ==================== FOOTER ====================
    st.markdown("---")
    st.markdown(f"""
    <div style="text-align: center; color: #666; padding: 1rem;">
        <strong>Climate Trend Analyzer</strong> | Data Science Project<br>
        Data Period: {year_range[0]} - {year_range[1]} | 
        Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br>
        <small>Interactive Dashboard - Filter, Explore, and Analyze Climate Trends</small>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()