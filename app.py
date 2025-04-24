import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from data_loader import load_and_preprocess_data
from data_analysis import perform_eda, get_key_insights, calculate_element_year_statistics, prepare_country_comparison_data
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
            comparison_data = prepare_country_comparison_data(filtered_data, selected_countries)
            if not comparison_data.empty:
                plot_comparisons(comparison_data, selected_countries)
            else:
                st.warning("No comparison data available. Ensure the dataset contains 'Country', 'Element', and 'Value' or 'Production' columns.")
            
            st.subheader("Key Insights")
            insights = get_key_insights(filtered_data)
            for insight in insights:
                st.write(f"• {insight}")
            
            st.subheader("Element-Year Statistics")
            element_year_stats = calculate_element_year_statistics(filtered_data)
            if not element_year_stats.empty:
                st.dataframe(element_year_stats.style.format({
                    'sum': '{:,.2f}',
                    'mean': '{:,.2f}',
                    'std': '{:,.2f}'
                }))
            else:
                st.warning("No element-year statistics available. Ensure the dataset contains 'Element', 'Year', and numerical columns like 'Production' or 'Value'.")
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
            features = st.multiselect("Select features", numeric_cols, default=['Country Code', 'Element Code', 'Item Code'])
            target_options = [col for col in numeric_cols if col not in features]
            
            if target_options:
                target = st.selectbox("Select target variable", target_options, index=target_options.index('Value') if 'Value' in target_options else 0)
                model_type = st.selectbox("Select model type", ["Linear Regression", "Random Forest", "XGBoost"])

                if features and target:
                    train_button = st.button("Train Model")
                    if train_button:
                        progress = st.progress(0)
                        with st.spinner("Training in progress..."):
                            model, metrics, test_data = train_model(data, features, target, model_type, progress_callback=progress.progress)
                        if model is not None:
                            st.success("Model trained successfully!")
                            st.session_state['model'] = model
                            st.session_state['test_data'] = test_data
                            st.session_state['features'] = features
                            st.session_state['target'] = target

                            st.subheader("Model Performance")
                            st.write(f"R² Score: {metrics['r2']:.4f}")
                            st.write(f"MAE: {metrics['mae']:.4f}")
                            st.write(f"RMSE: {metrics['rmse']:.4f}")
                        else:
                            st.error("Model training failed. Check the terminal logs for details.")

                    # Prediction input form
                    if 'model' in st.session_state and st.session_state['model'] is not None:
                        st.subheader("Make a Prediction")
                        user_input = {}
                        for f in st.session_state['features']:
                            if f in ['Country Code', 'Element Code', 'Item Code']:
                                valid_values = sorted(data[f].unique().astype(str))
                                user_input[f] = float(st.selectbox(f"Select {f}", valid_values, key=f))
                            else:
                                default_val = float(data[f].mean()) if f in data.columns else 0.0
                                user_input[f] = st.number_input(f"Enter value for {f}", value=default_val, key=f)

                        if st.button("Predict"):
                            try:
                                input_df = pd.DataFrame([user_input])
                                prediction = predict(st.session_state['model'], input_df)
                                st.success(f"Predicted {st.session_state['target']}: {prediction:.2f}")

                                # Plot prediction
                                st.subheader("Prediction Visualization")
                                test_data = st.session_state['test_data']
                                if 'X_test' in test_data and 'y_test' in test_data and 'y_pred' in test_data:
                                    # Create a DataFrame for plotting
                                    plot_data = pd.DataFrame({
                                        'Actual': test_data['y_test'],
                                        'Predicted': test_data['y_pred']
                                    }).reset_index(drop=True)
                                    # Add user prediction
                                    user_pred_df = pd.DataFrame({'Actual': [None], 'Predicted': [prediction]})
                                    plot_data = pd.concat([plot_data, user_pred_df], ignore_index=True)
                                    
                                    fig = px.scatter(plot_data, x=plot_data.index, y=['Actual', 'Predicted'],
                                                    labels={'value': st.session_state['target'], 'index': 'Sample'},
                                                    title=f'Actual vs Predicted {st.session_state['target']} (Last Point: User Prediction)')
                                    fig.update_traces(marker=dict(size=8), selector=dict(name='Actual'))
                                    fig.update_traces(marker=dict(size=8, symbol='x'), selector=dict(name='Predicted'))
                                    fig.add_scatter(x=[len(plot_data)-1], y=[prediction], mode='markers', 
                                                   marker=dict(size=12, color='red', symbol='star'),
                                                   name='User Prediction')
                                    st.plotly_chart(fig, use_container_width=True)
                                else:
                                    st.warning("Test data not available for plotting.")
                            except ValueError as e:
                                st.error(f"Prediction failed: {e}")
                                st.info("Ensure input values are valid and match the training data format.")
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
                        st.warning("Only one historical data point available, resulting in a flat forecast. Consider using a dataset with multiple years for better results.")
                    st.write("Forecasted Values:")
                    st.write(forecast_df.tail())  
                    try:
                        plot_forecast(data, forecast_df, country, item, metric)
                    except Exception as e:
                        st.error(f"Error plotting forecast: {e}")
                        st.info("The forecast was generated but could not be plotted. Check if the historical data contains the selected metric.")
                else:
                    st.error("Forecast generation failed.")
                    st.info("Possible reasons: Insufficient data points (need at least 1 year), invalid country/item/metric combination, or missing data. Check the terminal logs for details.")

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