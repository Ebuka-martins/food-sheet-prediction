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

# Enhanced data loading with disaggregation control
@st.cache_data
def load_data(disaggregate=True):
    return load_and_preprocess_data(disaggregate_europe=disaggregate)

# Add disaggregation control to sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Data Overview", "Data Analysis", "Machine Learning", "Forecasting", "Recommendations"])

# New: European data handling option
if page in ["Data Overview", "Data Analysis", "Machine Learning", "Forecasting"]:
    disaggregate = st.sidebar.checkbox("Disaggregate European data", value=True,
                                      help="Split European aggregates into individual countries")

# Load data with selected disaggregation option
data = load_data(disaggregate=disaggregate if 'disaggregate' in locals() else True)

# --- Page 1: Data Overview ---
if page == "Data Overview":
    st.header("Data Overview")
    if data is None or data.empty:
        st.error("No data available. Please check if the dataset file exists in the 'data' directory and has the required columns (Country, Element, Year, Value).")
    else:
        if disaggregate:
            st.info("Showing country-level data (European aggregates disaggregated)")
        else:
            st.info("Showing original data with European aggregates")
            
        st.write("Unique Years in Dataset:", sorted(data['Year'].unique()))
        st.write("Unique Year Codes in Dataset:", sorted(data['Year Code'].unique()))
        st.write(data.head())
        st.write(f"Dataset shape: {data.shape}")
        st.subheader("Basic Statistics")
        st.write(data.describe())
        st.subheader("Data Distributions")
        plot_distributions(data)

# --- Page 2: Data Analysis ---
elif page == "Data Analysis":
    st.header("Data Analysis")
    if data is None or data.empty:
        st.error("No data available. Please check if the dataset file exists in the 'data' directory and has the required columns (Country, Element, Year, Value).")
    else:
        # Display dataset columns for debugging
        st.write("Dataset Columns:", data.columns.tolist())
        
        analysis_options = st.radio("Analysis level", ["Countries", "Regions"], horizontal=True)
        
        if analysis_options == "Countries":
            available_options = sorted(data['Country'].unique())
        else:
            regions = ['Europe'] + [c for c in data['Country'].unique() if c not in ['Germany', 'France', 'Italy', 'United Kingdom']]
            available_options = sorted(regions)
        
        selected_items = st.multiselect(
            f"Select {analysis_options.lower()}",
            available_options,
            default=available_options[:3]
        )
        
        if selected_items:
            if analysis_options == "Countries":
                filtered_data = data[data['Country'].isin(selected_items)]
            else:
                if 'Europe' in selected_items:
                    european_countries = ['Germany', 'France', 'Italy', 'United Kingdom']
                    europe_data = data[data['Country'].isin(european_countries)]
                    other_data = data[data['Country'].isin([c for c in selected_items if c != 'Europe'])]
                    filtered_data = pd.concat([europe_data, other_data])
                else:
                    filtered_data = data[data['Country'].isin(selected_items)]
            
            st.subheader("Food Production Over Time")
            value_col = 'Production' if 'Production' in filtered_data.columns else 'Value' if 'Value' in filtered_data.columns else None
            if 'Year' in filtered_data.columns and value_col:
                plot_time_series(filtered_data, selected_items, value_col)
            else:
                st.error("Cannot plot time series: Missing 'Year' or 'Production/Value' columns.")
            
            st.subheader(f"{analysis_options} Comparisons")
            if {'Country', 'Element', 'Value'}.issubset(filtered_data.columns):
                comparison_data = prepare_country_comparison_data(filtered_data, selected_items)
                if not comparison_data.empty:
                    plot_comparisons(comparison_data, selected_items)
                else:
                    st.warning("No comparison data available. Ensure dataset contains valid 'Element' values (e.g., 'Production').")
            else:
                st.error("Cannot plot comparisons: Missing 'Country', 'Element', or 'Value' columns.")
            
            st.subheader("Key Insights")
            insights = get_key_insights(filtered_data)
            for insight in insights:
                st.write(f"• {insight}")

# --- Page 3: Machine Learning ---
elif page == "Machine Learning":
    st.header("Machine Learning Models")
    if data is None or data.empty:
        st.error("No data available. Please check if the dataset file exists in the 'data' directory and has the required columns (Country, Element, Year, Value).")
    else:
        # Get numeric columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        # Feature selection
        with st.expander("Feature Selection", expanded=True):
            features = st.multiselect(
                "Select features", 
                numeric_cols,
                default=numeric_cols[:min(3, len(numeric_cols))]  # Select up to 3 numeric columns by default
            )
            
            # Get target options (excluding selected features)
            target_options = [col for col in numeric_cols if col not in features]
            if not target_options:
                st.warning("No valid target options available with the selected features. Please select fewer features or different columns.")
            else:
                # Set default target to 'Year Code' if available, else first option
                preferred_target = 'Year Code' if 'Year Code' in target_options else target_options[0]
                target = st.selectbox(
                    "Select target variable",
                    target_options,
                    index=target_options.index(preferred_target) if preferred_target in target_options else 0
                )
        
        # Model selection
        model_type = st.selectbox(
            "Select model type", 
            ["Linear Regression", "Random Forest Classifier", "Random Forest", "XGBoost"]
        )

        if features and target:
            train_button = st.button("Train Model")
            if train_button:
                progress = st.progress(0)
                with st.spinner("Training in progress..."):
                    model, metrics, test_data = train_model(
                        data, 
                        features, 
                        target, 
                        model_type, 
                        progress_callback=progress.progress
                    )
                
                if model is not None:
                    st.success("Model trained successfully!")
                    st.session_state['model'] = model
                    st.session_state['test_data'] = test_data
                    st.session_state['features'] = features
                    st.session_state['target'] = target

                    st.subheader("Model Performance")
                    if model_type == "Random Forest Classifier":
                        st.write(f"Accuracy: {metrics['accuracy']:.4f}")
                        st.write("Classification Report:")
                        st.write(metrics['classification_report'])
                    else:
                        st.write(f"R² Score: {metrics['r2']:.4f}")
                        st.write(f"MAE: {metrics['mae']:.4f}")
                        st.write(f"RMSE: {metrics['rmse']:.4f}")
                else:
                    st.error("Model training failed. Check the terminal logs for details.")

            # Prediction section
            if 'model' in st.session_state and st.session_state['model'] is not None:
                st.subheader("Make a Prediction")
                user_input = {}
                
                for feature in st.session_state['features']:
                    if feature in ['Country Code', 'Element Code', 'Item Code']:
                        unique_values = sorted(data[feature].unique().astype(str))
                        user_input[feature] = float(st.selectbox(
                            f"Select {feature}", 
                            unique_values,
                            index=min(10, len(unique_values)-1)
                        ))
                    else:
                        default_val = float(data[feature].mean()) if feature in data.columns else 0.0
                        user_input[feature] = st.number_input(
                            f"Enter value for {feature}", 
                            value=default_val,
                            key=f"pred_{feature}"
                        )
                if st.button("Predict"):
                    try:
                        input_df = pd.DataFrame([user_input])
                        st.write("Input for prediction:", input_df)
                        prediction = predict(st.session_state['model'], input_df)
                        
                        if st.session_state['target'] == 'Year Code':
                            formatted_pred = f"{int(round(prediction))}"  
                        else:
                            formatted_pred = f"{prediction:.2f}"
                        
                        st.success(f"Predicted {st.session_state['target']}: {formatted_pred}")
                        # Improved visualization
                        test_data = st.session_state['test_data']
                        if 'y_test' in test_data and 'y_pred' in test_data:
                            plot_df = pd.DataFrame({
                                'Type': ['Actual'] * len(test_data['y_test']) + ['Predicted'] * len(test_data['y_pred']),
                                'Value': np.concatenate([test_data['y_test'], test_data['y_pred']]),
                                'Index': np.concatenate([np.arange(len(test_data['y_test'])), 
                                                        np.arange(len(test_data['y_pred']))])
                            })
                            
                            user_point = pd.DataFrame({
                                'Type': ['Your Prediction'],
                                'Value': [prediction],
                                'Index': [len(plot_df)]
                            })
                            plot_df = pd.concat([plot_df, user_point])
                            
                            fig = px.scatter(
                                plot_df, 
                                x='Index', 
                                y='Value', 
                                color='Type',
                                color_discrete_map={
                                    'Actual': 'blue',
                                    'Predicted': 'red',
                                    'Your Prediction': 'green'
                                },
                                symbol='Type',
                                symbol_map={
                                    'Actual': 'circle',
                                    'Predicted': 'x',
                                    'Your Prediction': 'star'
                                },
                                title=f'Model Predictions vs Actuals (Your Prediction: {formatted_pred})'
                            )
                            
                            fig.update_traces(marker=dict(size=10))
                            fig.update_layout(
                                xaxis_title='Sample Index',
                                yaxis_title=st.session_state['target'],
                                showlegend=True
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                    except Exception as e:
                        st.error(f"Prediction failed: {str(e)}")
                        st.info("Please ensure all input values are valid numbers")
        else:
            st.warning("Please select at least one feature and a valid target variable.")

# --- Page 4: Forecasting ---
elif page == "Forecasting":
    st.header("Forecasting")
    if data is None or data.empty:
        st.error("No data available. Please check if the dataset file exists in the 'data' directory and has the required columns (Country, Element, Year, Value).")
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
    if data is None or data.empty:
        st.error("No data available. Please check if the dataset file exists in the 'data' directory and has the required columns (Country, Element, Year, Value).")
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
# st.sidebar.markdown("Created as part of a Machine Learning project")