import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from data_loader import load_and_preprocess_data
from data_analysis import perform_eda, get_key_insights
from ml_models import train_model, predict
from visualizations import (
    plot_distributions,
    plot_time_series,
    plot_comparisons,
    plot_forecast
)

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

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Data Overview", "Data Analysis", "Machine Learning", "Forecasting", "Recommendations"])

# --- Page 1: Data Overview ---
if page == "Data Overview":
    st.header("Data Overview")
    if data.empty:
        st.error("No data available.")
    else:
        st.write(data.head())
        st.write(f"Dataset shape: {data.shape}")
        st.subheader("Basic Statistics")
        st.write(data.describe())
        st.subheader("Data Distributions")
        plot_distributions(data)

# --- Page 2: Data Analysis ---
elif page == "Data Analysis":
    st.header("Data Analysis")
    if data.empty:
        st.error("No data available.")
    else:
        countries = sorted(data['Country'].unique())
        selected_countries = st.multiselect("Select countries", countries, default=countries[:3])
        
        if selected_countries:
            filtered_data = data[data['Country'].isin(selected_countries)]
            st.subheader("Food Production Over Time")
            plot_time_series(filtered_data, selected_countries, 'Production')
            
            st.subheader("Country Comparisons")
            plot_comparisons(filtered_data, selected_countries)
            
            st.subheader("Key Insights")
            insights = get_key_insights(filtered_data)
            for insight in insights:
                st.write(f"• {insight}")
        else:
            st.info("Please select at least one country.")

# --- Page 3: Machine Learning ---
elif page == "Machine Learning":
    st.header("Machine Learning Models")

    if data.empty:
        st.error("No data available.")
    else:
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) > 1:
            features = st.multiselect("Select features", numeric_cols, default=numeric_cols[:min(3, len(numeric_cols))])
            target_options = [col for col in numeric_cols if col not in features]
            
            if target_options:
                target = st.selectbox("Select target variable", target_options)
                model_type = st.selectbox("Select model type", ["Linear Regression", "Random Forest", "XGBoost"])

                if features and target:
                    train_button = st.button("Train Model")
                    if train_button:
                        progress = st.progress(0)
                        with st.spinner("Training in progress..."):
                            model, metrics, pipeline = train_model(data, features, target, model_type, progress_callback=progress.progress)
                        st.success("Model trained successfully!")

                        st.subheader("Model Performance")
                        st.write(f"R² Score: {metrics['r2']:.4f}")
                        st.write(f"MAE: {metrics['mae']:.4f}")
                        st.write(f"RMSE: {metrics['rmse']:.4f}")

                        # Prediction input form
                        st.subheader("Make a Prediction")
                        user_input = {}
                        for f in features:
                            default_val = float(data[f].mean()) if f in data.columns else 0.0
                            user_input[f] = st.number_input(f"Enter value for {f}", value=default_val)

                        if st.button("Predict"):
                            input_df = pd.DataFrame([user_input])
                            prediction = predict(pipeline, input_df)
                            st.success(f"Predicted {target}: {prediction:.2f}")
                else:
                    st.warning("Select features and target to proceed.")
            else:
                st.warning("Not enough numeric columns available.")
        else:
            st.error("Dataset lacks numeric columns.")

# --- Page 4: Forecasting ---
elif page == "Forecasting":
    st.header("Forecasting")
    if data.empty:
        st.error("No data available. Please check if the dataset loaded correctly.")
    else:
        # Use valid Element values from the dataset
        valid_metrics = sorted(data['Element'].unique())
        country = st.selectbox("Select Country", sorted(data['Country'].unique()))
        item = st.selectbox("Select Item", sorted(data['Item'].unique()))
        metric = st.selectbox("Select Metric", valid_metrics)

        if st.button("Generate Forecast"):
            with st.spinner("Generating forecast..."):
                from forecasting import generate_forecast
                forecast_df = generate_forecast(data, country, item, metric)

                if forecast_df is not None:
                    st.success("Forecast generated!")
                    if len(forecast_df['Forecast'].unique()) == 1:
                        st.warning("Only one historical data point available, resulting in a flat forecast. Consider " \
                        "using a dataset with multiple years for better results.")
                    st.write("Forecasted Values:")
                    st.write(forecast_df.tail())  # Show forecast output
                    try:
                        plot_forecast(data, forecast_df, country, item, metric)
                    except Exception as e:
                        st.error(f"Error plotting forecast: {e}")
                        st.info("The forecast was generated but could not be plotted. Check if the historical data " \
                        "contains the selected metric.")
                else:
                    st.error("Forecast generation failed.")
                    st.info("Possible reasons: Insufficient data points (need at least 1 year), invalid country/item/metric " \
                    "combination, or missing data. Check the terminal logs for details.")



# --- Page 5: Recommendations ---
elif page == "Recommendations":
    st.header("Data-Driven Recommendations")

    if data.empty:
        st.error("No data available.")
    else:
        st.subheader("Food Production Recommendations")
        st.write("1. Diversify agricultural production to enhance food security.")
        st.write("2. Reduce import dependency through strategic domestic policies.")
        st.write("3. Focus on resource efficiency to increase yield.")

        st.subheader("Food Security Insights")
        st.write("1. Import-reliant regions are vulnerable.")
        st.write("2. Enhance regional cooperation and food reserves.")
        st.write("3. Improve infrastructure to cut post-harvest losses.")

        st.subheader("Sustainability Recommendations")
        st.write("1. Promote eco-friendly farming practices.")
        st.write("2. Encourage dietary diversity for resilience.")
        st.write("3. Monitor climate impact on crop yield.")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("Created as part of a Machine Learning project")
