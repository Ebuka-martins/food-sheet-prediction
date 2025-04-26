import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from data_loader import EUROPEAN_COUNTRIES

# Constants for plot sizing
DEFAULT_HEIGHT = 500
TALL_HEIGHT = 700
MAP_HEIGHT = 600
SMALL_HEIGHT = 400

# Extract just the country names from EUROPEAN_COUNTRIES
EUROPEAN_COUNTRY_NAMES = [country[1] for country in EUROPEAN_COUNTRIES]

def get_plot_layout(title=None, height=DEFAULT_HEIGHT):
    """Returns a standardized layout configuration for plots"""
    return {
        "title": dict(text=title, x=0.5, xanchor='center', font=dict(size=18)),
        "height": height,
        "margin": dict(t=60, b=40, l=50, r=50),
        "template": "plotly_white",
        "legend": dict(title='', orientation="h", y=-0.2, x=0.5, xanchor='center'),
    }

def plot_distributions(df, columns=None):
    """Plot histograms for numeric columns"""
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
        columns = [col for col in columns if 'Year' not in col]

    if len(columns) > 6:
        columns = columns[:6]

    for col in columns:
        try:
            fig = px.histogram(df, x=col)
            fig.update_layout(**get_plot_layout(f"Distribution of {col}", height=SMALL_HEIGHT))
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error plotting distribution for {col}: {e}")

def plot_time_series(df, countries=None, metric='Production'):
    """Plot time series data with European country handling"""
    if 'Year' not in df.columns or metric not in df.columns:
        st.error(f"Required columns not found: 'Year' or '{metric}'")
        return

    # Handle Europe selection
    if countries is not None and 'Europe' in countries:
        countries = [c for c in countries if c != 'Europe'] + EUROPEAN_COUNTRY_NAMES
    
    if countries is None and 'Country' in df.columns:
        top_countries = df.groupby('Country')[metric].sum().sort_values(ascending=False).head(10).index.tolist()
        df_filtered = df[df['Country'].isin(top_countries)]
    elif countries is not None and 'Country' in df.columns:
        df_filtered = df[df['Country'].isin(countries)]
    else:
        df_filtered = df

    if 'Country' in df.columns:
        df_grouped = df_filtered.groupby(['Year', 'Country'])[metric].sum().reset_index()
        
        # Special handling for European countries
        if set(countries or []).intersection(set(EUROPEAN_COUNTRY_NAMES)):
            europe_total = df_grouped[df_grouped['Country'].isin(EUROPEAN_COUNTRY_NAMES)]\
                .groupby('Year')[metric].sum().reset_index()
            europe_total['Country'] = 'Europe (Total)'
            df_grouped = pd.concat([df_grouped, europe_total])
            
        fig = px.line(df_grouped, x='Year', y=metric, color='Country',
                     color_discrete_sequence=px.colors.qualitative.Plotly)
    else:
        df_grouped = df_filtered.groupby('Year')[metric].sum().reset_index()
        fig = px.line(df_grouped, x='Year', y=metric)

    fig.update_layout(**get_plot_layout(
        f"{metric} Over Time by Country" if 'Country' in df.columns else f"{metric} Over Time",
        height=TALL_HEIGHT if 'Country' in df.columns else DEFAULT_HEIGHT
    ))
    st.plotly_chart(fig, use_container_width=True)

def plot_comparisons(df, countries=None):
    """
    Plot comparisons for selected countries or regions using Element-based metrics.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with columns Country and metrics (e.g., Production, Import Quantity)
    countries : list
        List of countries or regions
    """
    if df.empty or 'Country' not in df.columns:
        st.error("No data available for comparison. Ensure dataset contains 'Country' column.")
        return
    
    # Select metrics (exclude 'Country')
    metrics = [col for col in df.columns if col != 'Country']
    if not metrics:
        st.error("No metrics available for comparison. Ensure dataset contains 'Element' values like 'Production'.")
        return
    
    # Filter for selected countries
    if countries:
        df = df[df['Country'].isin(countries)]
    
    # Handle Europe aggregation
    if countries and set(countries).intersection(set(EUROPEAN_COUNTRY_NAMES)):
        europe_data = df[df['Country'].isin(EUROPEAN_COUNTRY_NAMES)]
        if not europe_data.empty:
            europe_totals = europe_data[metrics].sum().to_frame().T
            europe_totals['Country'] = 'Europe (Total)'
            df = pd.concat([df[~df['Country'].isin(EUROPEAN_COUNTRY_NAMES)], europe_totals])
    
    # Melt data for plotting
    plot_data = df.melt(id_vars='Country', value_vars=metrics, 
                        var_name='Metric', value_name='Value')
    
    # Create bar plot
    fig = px.bar(
        plot_data,
        x='Country',
        y='Value',
        color='Metric',
        barmode='group',
        title='Country Comparisons by Metric',
        text_auto='.2s',
        color_discrete_sequence=px.colors.qualitative.Plotly
    )
    
    fig.update_layout(**get_plot_layout('Country Comparison by Key Metrics', height=TALL_HEIGHT))
    st.plotly_chart(fig, use_container_width=True)

def plot_food_balance(df, country):
    """Plot food balance composition for a specific country"""
    if not all(col in df.columns for col in ['Country', 'Food', 'Feed', 'Seed', 'Losses']):
        st.error("Required columns not found for food balance chart")
        return

    df_country = df[df['Country'] == country]

    if 'Year' in df.columns:
        df_agg = df_country.groupby('Year')[['Food', 'Feed', 'Seed', 'Losses']].sum().reset_index()
        df_agg = df_agg.melt(id_vars=['Year'], value_vars=['Food', 'Feed', 'Seed', 'Losses'],
                             var_name='Category', value_name='Value')
        fig = px.area(df_agg, x='Year', y='Value', color='Category')
        fig.update_layout(**get_plot_layout(f'Food Balance Composition for {country}'))
    else:
        df_agg = pd.DataFrame({
            'Category': ['Food', 'Feed', 'Seed', 'Losses'],
            'Value': [
                df_country['Food'].sum(),
                df_country['Feed'].sum(),
                df_country['Seed'].sum(),
                df_country['Losses'].sum()
            ]
        })
        fig = px.pie(df_agg, values='Value', names='Category')
        fig.update_layout(**get_plot_layout(f'Food Balance Composition for {country}'))

    st.plotly_chart(fig, use_container_width=True)

def plot_map(df, metric='Production', year=None):
    """Plot choropleth map with European aggregation"""
    if not all(col in df.columns for col in ['Country', metric]):
        st.error(f"Required columns not found: 'Country' or '{metric}'")
        return

    if year is not None and 'Year' in df.columns:
        df_filtered = df[df['Year'] == year]
    elif 'Year' in df.columns:
        latest_year = df['Year'].max()
        df_filtered = df[df['Year'] == latest_year]
    else:
        df_filtered = df

    # Aggregate European countries if needed
    if 'Europe' in df_filtered['Country'].unique():
        europe_data = df_filtered[df_filtered['Country'].isin(EUROPEAN_COUNTRY_NAMES)]
        europe_agg = europe_data.groupby('Item')[metric].sum().reset_index()
        europe_agg['Country'] = 'Europe'
        df_filtered = pd.concat([df_filtered[~df_filtered['Country'].isin(EUROPEAN_COUNTRY_NAMES)], europe_agg])

    df_agg = df_filtered.groupby('Country')[metric].sum().reset_index()

    fig = px.choropleth(df_agg, locations='Country', locationmode='country names',
                        color=metric, hover_name='Country',
                        color_continuous_scale=px.colors.sequential.Viridis)
    fig.update_layout(**get_plot_layout(f'{metric} by Country', height=MAP_HEIGHT))
    st.plotly_chart(fig, use_container_width=True)

def plot_correlation_matrix(df, columns=None):
    """Plot correlation matrix for numeric columns"""
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
        columns = [col for col in columns if col.lower() not in ['year', 'id']]

    corr_matrix = df[columns].corr()

    fig = px.imshow(corr_matrix, text_auto=True, aspect="auto")
    fig.update_layout(**get_plot_layout('Correlation Matrix', height=TALL_HEIGHT))
    st.plotly_chart(fig, use_container_width=True)

def plot_forecast(historical_data, forecast_data, country, item, metric):
    """
    Plot historical and forecasted data.
    
    Args:
        historical_data (pd.DataFrame): Historical data with Year, Value, Country, Item, Element
        forecast_data (pd.DataFrame): Forecast data with Year, Forecast
        country (str): Selected country
        item (str): Selected item
        metric (str): Selected metric (Element)
    """
    try:
        hist_filtered = historical_data[
            (historical_data['Country'] == country) &
            (historical_data['Item'] == item) &
            (historical_data['Element'] == metric)
        ][['Year', 'Value']].copy()

        if hist_filtered.empty:
            st.error(f"No historical data found for {country}, {item}, {metric}")
            return

        hist_filtered['Year'] = hist_filtered['Year'].astype(int)

        if 'Year' not in forecast_data.columns or 'Forecast' not in forecast_data.columns:
            st.error("Required columns not found in forecast data: 'Year' or 'Forecast'")
            return

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=hist_filtered['Year'],
            y=hist_filtered['Value'],
            mode='lines+markers',
            name='Historical',
            line=dict(color='blue')
        ))

        fig.add_trace(go.Scatter(
            x=forecast_data['Year'],
            y=forecast_data['Forecast'],
            mode='lines+markers',
            name='Forecast',
            line=dict(color='red', dash='dash')
        ))

        fig.update_layout(
            **get_plot_layout(f'{metric} Forecast for {item} in {country}', height=DEFAULT_HEIGHT),
            xaxis_title='Year',
            yaxis_title=metric
        )

        st.plotly_chart(fig, use_container_width=True)
    
    except Exception as e:
        st.error(f"Error plotting forecast: {e}")
        st.info(f"Ensure historical data contains 'Year' and 'Value' for {metric}")