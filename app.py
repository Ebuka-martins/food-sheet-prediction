import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from data_loader import load_and_preprocess_data
from data_analysis import perform_eda, get_key_insights
from ml_models import train_model, predict
from visualizations import plot_distributions, plot_time_series, plot_comparisons

# Page configuration
st.set_page_config(
    page_title="Food Balance Sheet Analysis",
    page_icon="🍲",
    layout="wide"
)

# Title and introduction
st.title("Food Balance Sheet Analysis - Europe")
st.markdown("""
This application analyzes food balance sheet data for European countries,
provides insights into food production and consumption patterns, and offers
predictive analytics using machine learning models.
""")

# Load data
@st.cache_data
def load_data():
    return load_and_preprocess_data()

data = load_data()

# Sidebar for navigation and options
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Data Overview", "Data Analysis", "Machine Learning", "Recommendations"])

# Data Overview page
if page == "Data Overview":
    st.header("Data Overview")
    
    if data.empty:
        st.error("No data available. Please check if the data file exists and is properly formatted.")
    else:
        st.write(data.head())
        st.write(f"Dataset shape: {data.shape}")
        
        # Basic statistics
        st.subheader("Basic Statistics")
        st.write(data.describe())
        
        # Data distributions
        st.subheader("Data Distributions")
        plot_distributions(data)

# Data Analysis page
elif page == "Data Analysis":
    st.header("Data Analysis")
    
    if data.empty:
        st.error("No data available. Please check if the data file exists and is properly formatted.")
    else:
        # Country selection
        if 'Country' in data.columns:
            countries = sorted(data['Country'].unique())
            selected_countries = st.multiselect("Select countries to analyze", countries, default=countries[:3] if len(countries) >= 3 else countries)
            
            if selected_countries:
                # Filter data
                filtered_data = data[data['Country'].isin(selected_countries)]
                
                # Time series analysis
                st.subheader("Food Production Over Time")
                if 'Production' in data.columns:
                    plot_time_series(filtered_data, selected_countries, 'Production')
                else:
                    numeric_cols = filtered_data.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 0:
                        selected_metric = st.selectbox("Select metric for time series", numeric_cols)
                        plot_time_series(filtered_data, selected_countries, selected_metric)
                    else:
                        st.error("No numeric columns available for time series analysis")
                
                # Comparisons
                st.subheader("Country Comparisons")
                plot_comparisons(filtered_data, selected_countries)
                
                # Key insights
                st.subheader("Key Insights")
                insights = get_key_insights(filtered_data)
                for insight in insights:
                    st.write(f"• {insight}")
            else:
                st.info("Please select at least one country to analyze")
        else:
            st.error("Country column not found in the dataset")

# Machine Learning page
elif page == "Machine Learning":
    st.header("Machine Learning Models")
    
    if data.empty:
        st.error("No data available. Please check if the data file exists and is properly formatted.")
    else:
        # Get numeric columns for features and target
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) > 1:
            # Feature and target selection
            features = st.multiselect("Select features", numeric_cols, default=numeric_cols[:min(3, len(numeric_cols))])
            target_options = [col for col in numeric_cols if col not in features]
            
            if target_options:
                target = st.selectbox("Select target variable", target_options)
                
                if features and target:
                    # Model selection
                    model_type = st.selectbox("Select model type", ["Linear Regression", "Random Forest", "XGBoost"])
                    
                    # Train model button
                    if st.button("Train Model"):
                        with st.spinner("Training model..."):
                            model, metrics, _ = train_model(data, features, target, model_type)
                            
                            if model:
                                # Display metrics
                                st.subheader("Model Performance")
                                st.write(f"R² Score: {metrics['r2']:.4f}")
                                st.write(f"MAE: {metrics['mae']:.4f}")
                                st.write(f"RMSE: {metrics['rmse']:.4f}")
                                
                                # Predictions
                                st.subheader("Predictions vs Actual")
                                st.info("This would show a visualization of predictions vs actual values")
                            else:
                                st.error("Error training model. Please check the console for more information.")
                else:
                    st.warning("Please select features and a target variable")
            else:
                st.warning("Not enough numeric columns to separate features and target")
        else:
            st.error("Not enough numeric columns for training a model")

# Recommendations page
elif page == "Recommendations":
    st.header("Data-Driven Recommendations")
    
    if data.empty:
        st.error("No data available. Please check if the data file exists and is properly formatted.")
    else:
        # Generate recommendations based on data analysis and ML insights
        st.subheader("Food Production Recommendations")
        st.write("1. Based on the analysis of trends in food consumption patterns, diversifying agricultural production could enhance food security.")
        st.write("2. Countries with high import dependency ratios could benefit from increasing domestic production of key food items.")
        st.write("3. Our predictive models suggest focusing on efficient use of agricultural resources to maximize yields and reduce losses.")
        
        st.subheader("Food Security Insights")
        st.write("1. The data indicates potential food security challenges in regions with high import dependency.")
        st.write("2. To address these challenges, we recommend strengthening regional cooperation and trade networks.")
        st.write("3. Investing in storage infrastructure can help reduce post-harvest losses and improve overall food availability.")
        
        # Additional custom recommendations
        st.subheader("Sustainability Recommendations")
        st.write("1. Balance between production and environmental sustainability should be a priority.")
        st.write("2. Diversification of food sources can increase resilience against climate and market shocks.")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("Created as part of a Machine Learning project")