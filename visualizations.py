import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_distributions(df, columns=None):
    """
    Plot distributions of numeric columns
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    columns : list
        List of columns to plot (if None, use all numeric columns)
    """
    if columns is None:
        # Select numeric columns with reasonable ranges (exclude years)
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
        columns = [col for col in columns if 'Year' not in col]
    
    # Limit to first 6 columns to avoid overwhelming visuals
    if len(columns) > 6:
        columns = columns[:6]
    
    for col in columns:
        try:
            fig = px.histogram(df, x=col, title=f"Distribution of {col}")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error plotting distribution for {col}: {e}")

def plot_time_series(df, countries=None, metric='Production'):
    """
    Plot time series data for selected countries
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    countries : list
        List of countries to plot
    metric : str
        Metric to plot
    """
    if 'Year' not in df.columns or metric not in df.columns:
        st.error(f"Required columns not found: 'Year' or '{metric}'")
        return
    
    if countries is None and 'Country' in df.columns:
        # Use all countries, but limit to top 10 by metric
        top_countries = df.groupby('Country')[metric].sum().sort_values(ascending=False).head(10).index.tolist()
        df_filtered = df[df['Country'].isin(top_countries)]
    elif countries is not None and 'Country' in df.columns:
        df_filtered = df[df['Country'].isin(countries)]
    else:
        df_filtered = df
    
    # Group by year and country
    if 'Country' in df.columns:
        df_grouped = df_filtered.groupby(['Year', 'Country'])[metric].sum().reset_index()
        fig = px.line(df_grouped, x='Year', y=metric, color='Country', 
                      title=f"{metric} Over Time by Country")
    else:
        df_grouped = df_filtered.groupby('Year')[metric].sum().reset_index()
        fig = px.line(df_grouped, x='Year', y=metric, 
                      title=f"{metric} Over Time")
    
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

def plot_comparisons(df, countries=None, metrics=None):
    """
    Plot country comparisons for selected metrics
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    countries : list
        List of countries to compare
    metrics : list
        List of metrics to compare
    """
    if 'Country' not in df.columns:
        st.error("Required column not found: 'Country'")
        return
    
    if countries is None:
        # Use top 10 countries by Production
        if 'Production' in df.columns:
            countries = df.groupby('Country')['Production'].sum().sort_values(ascending=False).head(10).index.tolist()
        else:
            countries = df['Country'].unique().tolist()[:10]
    
    if metrics is None:
        # Use common food balance sheet metrics
        potential_metrics = ['Production', 'Import Quantity', 'Export Quantity', 'Food', 'Feed', 'Losses']
        metrics = [m for m in potential_metrics if m in df.columns][:3]  # Limit to first 3
    
    # Filter data
    df_filtered = df[df['Country'].isin(countries)]
    
    # Aggregate by country
    df_agg = df_filtered.groupby('Country')[metrics].sum().reset_index()
    
    # Create grouped bar chart
    fig = px.bar(df_agg, x='Country', y=metrics, barmode='group',
                 title='Country Comparison by Key Metrics')
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

def plot_food_balance(df, country):
    """
    Plot food balance composition for a selected country
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    country : str
        Country to plot
    """
    if not all(col in df.columns for col in ['Country', 'Food', 'Feed', 'Seed', 'Losses']):
        st.error("Required columns not found for food balance chart")
        return
    
    # Filter for the selected country
    df_country = df[df['Country'] == country]
    
    # Group by year
    if 'Year' in df.columns:
        df_agg = df_country.groupby('Year')[['Food', 'Feed', 'Seed', 'Losses']].sum().reset_index()
        df_agg = df_agg.melt(id_vars=['Year'], value_vars=['Food', 'Feed', 'Seed', 'Losses'],
                            var_name='Category', value_name='Value')
        
        fig = px.area(df_agg, x='Year', y='Value', color='Category',
                    title=f'Food Balance Composition for {country}')
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
        
        fig = px.pie(df_agg, values='Value', names='Category',
                   title=f'Food Balance Composition for {country}')
    
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

def plot_map(df, metric='Production', year=None):
    """
    Plot a choropleth map showing a metric by country
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    metric : str
        Metric to plot
    year : int
        Year to filter data by (if None, use the latest year)
    """
    if not all(col in df.columns for col in ['Country', metric]):
        st.error(f"Required columns not found: 'Country' or '{metric}'")
        return
    
    # Filter by year if specified
    if year is not None and 'Year' in df.columns:
        df_filtered = df[df['Year'] == year]
    elif 'Year' in df.columns:
        # Use the latest year
        latest_year = df['Year'].max()
        df_filtered = df[df['Year'] == latest_year]
    else:
        df_filtered = df
    
    # Aggregate by country
    df_agg = df_filtered.groupby('Country')[metric].sum().reset_index()
    
    # Create choropleth map
    fig = px.choropleth(df_agg, locations='Country', locationmode='country names',
                       color=metric, hover_name='Country',
                       color_continuous_scale=px.colors.sequential.Viridis,
                       title=f'{metric} by Country')
    
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)

def plot_correlation_matrix(df, columns=None):
    """
    Plot correlation matrix for numeric columns
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    columns : list
        List of columns to include in correlation matrix
    """
    if columns is None:
        # Use numeric columns
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
        # Exclude year and ID columns
        columns = [col for col in columns if col.lower() not in ['year', 'id']]
    
    # Calculate correlation matrix
    corr_matrix = df[columns].corr()
    
    # Create heatmap
    fig = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                   title='Correlation Matrix')
    fig.update_layout(height=700)
    st.plotly_chart(fig, use_container_width=True)

def plot_forecast(historical_data, forecast_data, country, item, metric='Production'):
    """
    Plot historical data and forecast
    
    Parameters:
    -----------
    historical_data : pd.DataFrame
        Historical data
    forecast_data : pd.DataFrame
        Forecast data
    country : str
        Country name
    item : str
        Item name
    metric : str
        Metric to plot
    """
    if 'Year' not in historical_data.columns or metric not in historical_data.columns:
        st.error(f"Required columns not found in historical data: 'Year' or '{metric}'")
        return
    
    if 'Year' not in forecast_data.columns or f'Forecasted_{metric}' not in forecast_data.columns:
        st.error(f"Required columns not found in forecast data: 'Year' or 'Forecasted_{metric}'")
        return
    
    # Filter historical data
    hist_filtered = historical_data[
        (historical_data['Country'] == country) & 
        (historical_data['Item'] == item)
    ]
    
    # Sort by year
    hist_filtered = hist_filtered.sort_values('Year')
    
    # Create figure
    fig = go.Figure()
    
    # Add historical data
    fig.add_trace(go.Scatter(
        x=hist_filtered['Year'],
        y=hist_filtered[metric],
        mode='lines+markers',
        name='Historical',
        line=dict(color='blue')
    ))
    
    # Add forecast data
    fig.add_trace(go.Scatter(
        x=forecast_data['Year'],
        y=forecast_data[f'Forecasted_{metric}'],
        mode='lines+markers',
        name='Forecast',
        line=dict(color='red', dash='dash')
    ))
    
    # Update layout
    fig.update_layout(
        title=f'{metric} Forecast for {item} in {country}',
        xaxis_title='Year',
        yaxis_title=metric,
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)