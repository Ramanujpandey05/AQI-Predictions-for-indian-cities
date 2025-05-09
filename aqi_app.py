import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image

# Set page config
st.set_page_config(
    page_title="AQI Prediction for Indian Cities",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load and preprocess data
@st.cache_data
def load_data():
    df = pd.read_csv('city_day.csv')
    df['BTX'] = df['Benzene'] + df['Toluene'] + df['Xylene']
    df['Particulate_Matter'] = df['PM2.5'] + df['PM10']
    df.drop(['Benzene', 'Toluene', 'Xylene', 'PM2.5', 'PM10'], axis=1, inplace=True)
    df = df.dropna(subset=['AQI'])
    return df

df = load_data()

# Train model
@st.cache_resource
def train_model():
    X = df[['NO2', 'CO', 'SO2', 'O3', 'BTX', 'Particulate_Matter']]
    y = df['AQI']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model, X_test, y_test

model, X_test, y_test = train_model()

# AQI Bucket classification
def get_aqi_bucket(aqi):
    if pd.isna(aqi):
        return np.nan
    if aqi <= 50:
        return 'Good'
    elif aqi <= 100:
        return 'Satisfactory'
    elif aqi <= 200:
        return 'Moderate'
    elif aqi <= 300:
        return 'Poor'
    elif aqi <= 400:
        return 'Very Poor'
    else:
        return 'Severe'

# Streamlit app
def main():
    st.title("🌫️ Air Quality Index (AQI) Prediction for Indian Cities")
    st.markdown("""
    This app predicts the Air Quality Index (AQI) based on pollutant levels in Indian cities.
    """)
    
    # Sidebar
    st.sidebar.header("About")
    st.sidebar.info("""
    This app uses a Random Forest model trained on historical air quality data from Indian cities to predict AQI.
    Adjust the pollutant levels using the sliders to see how they affect the AQI prediction.
    """)
    
    st.sidebar.header("Model Performance")
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    st.sidebar.metric("RMSE", f"{rmse:.2f}")
    st.sidebar.metric("R² Score", f"{r2:.2f}")
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["AQI Prediction", "Data Visualization", "About AQI"])
    
    with tab1:
        st.header("AQI Prediction")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # City selection
            cities = sorted(df['City'].unique())
            selected_city = st.selectbox("Select City", cities)
            
            # Pollutant sliders
            st.subheader("Enter Pollutant Levels")
            no2 = st.slider("NO2 (µg/m³)", min_value=0.0, max_value=200.0, value=40.0, step=0.1)
            co = st.slider("CO (µg/m³)", min_value=0.0, max_value=500.0, value=100.0, step=1.0)
            so2 = st.slider("SO2 (µg/m³)", min_value=0.0, max_value=200.0, value=20.0, step=0.1)
            o3 = st.slider("O3 (µg/m³)", min_value=0.0, max_value=200.0, value=50.0, step=0.1)
            btx = st.slider("BTX (Benzene+Toluene+Xylene) (µg/m³)", min_value=0.0, max_value=50.0, value=5.0, step=0.1)
            pm = st.slider("Particulate Matter (PM2.5 + PM10) (µg/m³)", min_value=0.0, max_value=500.0, value=100.0, step=1.0)
            
            if st.button("Predict AQI"):
                input_data = pd.DataFrame([[no2, co, so2, o3, btx, pm]], 
                                        columns=['NO2', 'CO', 'SO2', 'O3', 'BTX', 'Particulate_Matter'])
                aqi_pred = model.predict(input_data)[0]
                aqi_bucket = get_aqi_bucket(aqi_pred)
                
                st.success(f"Predicted AQI: {aqi_pred:.1f}")
                st.markdown(f"### AQI Category: **{aqi_bucket}**")
                
                # AQI color indicator
                colors = {
                    'Good': '#55A84F',
                    'Satisfactory': '#A3C853',
                    'Moderate': '#FFF833',
                    'Poor': '#F29C33',
                    'Very Poor': '#E93F33',
                    'Severe': '#AF2D24'
                }
                
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = aqi_pred,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "AQI Level"},
                    gauge = {
                        'axis': {'range': [0, 500]},
                        'steps': [
                            {'range': [0, 50], 'color': colors['Good']},
                            {'range': [50, 100], 'color': colors['Satisfactory']},
                            {'range': [100, 200], 'color': colors['Moderate']},
                            {'range': [200, 300], 'color': colors['Poor']},
                            {'range': [300, 400], 'color': colors['Very Poor']},
                            {'range': [400, 500], 'color': colors['Severe']}
                        ],
                        'bar': {'color': "black"}
                    }
                ))
                
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("City Pollution Trends")
            
            # Filter data for selected city
            city_data = df[df['City'] == selected_city].copy()
            city_data['Date'] = pd.to_datetime(city_data['Date'])
            city_data['Year'] = city_data['Date'].dt.year
            city_data['Month'] = city_data['Date'].dt.month
            
            # Yearly AQI trend
            yearly_avg = city_data.groupby('Year')['AQI'].mean().reset_index()
            fig1 = px.line(yearly_avg, x='Year', y='AQI', 
                          title=f'Yearly Average AQI Trend for {selected_city}',
                          markers=True)
            fig1.update_layout(yaxis_title="Average AQI")
            st.plotly_chart(fig1, use_container_width=True)
            
            # Monthly AQI trend
            monthly_avg = city_data.groupby('Month')['AQI'].mean().reset_index()
            fig2 = px.line(monthly_avg, x='Month', y='AQI', 
                          title=f'Monthly Average AQI Pattern for {selected_city}',
                          markers=True)
            fig2.update_layout(yaxis_title="Average AQI")
            st.plotly_chart(fig2, use_container_width=True)
    
    with tab2:
        st.header("Data Visualization")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Pollutant Distribution")
            pollutant = st.selectbox("Select Pollutant", 
                                   ['NO2', 'CO', 'SO2', 'O3', 'BTX', 'Particulate_Matter'])
            
            fig = px.histogram(df, x=pollutant, nbins=50, 
                              title=f'Distribution of {pollutant} Levels')
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            st.subheader("AQI Distribution by City")
            top_n = st.slider("Number of Cities to Show", 5, 20, 10)
            
            city_avg = df.groupby('City')['AQI'].mean().sort_values(ascending=False).reset_index().head(top_n)
            fig = px.bar(city_avg, x='City', y='AQI', 
                         title=f'Top {top_n} Cities by Average AQI')
            st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Pollutant Correlation Matrix")
        corr_matrix = df[['NO2', 'CO', 'SO2', 'O3', 'BTX', 'Particulate_Matter', 'AQI']].corr()
        fig = px.imshow(corr_matrix,
                        labels=dict(x="Pollutants", y="Pollutants", color="Correlation"),
                        x=corr_matrix.columns,
                        y=corr_matrix.columns,
                        color_continuous_scale='RdBu_r',
                        zmin=-1, zmax=1,
                        title="Correlation Between Pollutants and AQI")
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.header("About Air Quality Index (AQI)")
        
        st.markdown("""
        ### What is AQI?
        The Air Quality Index (AQI) is a numerical scale used to report daily air quality. 
        It provides information about how clean or polluted the air is and what associated 
        health effects might be a concern.
        """)
        
        st.subheader("AQI Categories and Health Impacts")
        
        aqi_table = pd.DataFrame({
            'AQI Range': ['0-50', '51-100', '101-200', '201-300', '301-400', '401-500'],
            'Category': ['Good', 'Satisfactory', 'Moderate', 'Poor', 'Very Poor', 'Severe'],
            'Health Impact': [
                'Minimal impact',
                'Minor breathing discomfort to sensitive people',
                'Breathing discomfort to people with lung disease and discomfort to people with heart disease',
                'Breathing discomfort to most people on prolonged exposure',
                'Respiratory illness on prolonged exposure',
                'Affects healthy people and seriously impacts those with existing diseases'
            ]
        })
        
        st.table(aqi_table)
        
        st.subheader("Key Pollutants")
        
        pollutants_info = pd.DataFrame({
            'Pollutant': ['PM2.5', 'PM10', 'NO2', 'CO', 'SO2', 'O3', 'BTX'],
            'Description': [
                'Fine particulate matter (≤2.5µm) - can penetrate deep into lungs',
                'Coarse particulate matter (≤10µm) - can irritate eyes, nose and throat',
                'Nitrogen dioxide - can cause respiratory problems',
                'Carbon monoxide - reduces oxygen delivery to body organs',
                'Sulfur dioxide - can affect respiratory system and lung function',
                'Ozone - can cause breathing problems, trigger asthma',
                'Benzene+Toluene+Xylene - volatile organic compounds with various health effects'
            ],
            'Major Sources': [
                'Vehicle emissions, power plants, wildfires',
                'Dust, pollen, mold, construction sites',
                'Vehicle emissions, power plants',
                'Incomplete combustion of fuels',
                'Burning of fossil fuels by power plants',
                'Chemical reactions between pollutants in sunlight',
                'Vehicle emissions, industrial processes, paints'
            ]
        })
        
        st.table(pollutants_info)
        
        st.markdown("""
        ### How to Use This App
        1. Select a city from the dropdown
        2. Adjust the pollutant levels using the sliders
        3. Click "Predict AQI" to see the predicted AQI value and category
        4. Explore the visualizations to understand pollution trends and patterns
        """)

if __name__ == "__main__":
    main()