import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


DEFAULT_HEIGHT = 500
TALL_HEIGHT = 700
MAP_HEIGHT = 600
SMALL_HEIGHT = 400


def get_plot_layout(title=None, height=DEFAULT_HEIGHT):
    return {
        "title": dict(text=title, x=0.5, xanchor='center', font=dict(size=18)),
        "height": height,
        "margin": dict(t=60, b=40, l=50, r=50),
        "template": "plotly_white",
        "legend": dict(title='', orientation="h", y=-0.2, x=0.5, xanchor='center'),
    }

# === Plot Functions ===
def plot_distributions(df, columns=None):
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
    if 'Year' not in df.columns or metric not in df.columns:
        st.error(f"Required columns not found: 'Year' or '{metric}'")
        return

    if countries is None and 'Country' in df.columns:
        top_countries = df.groupby('Country')[metric].sum().sort_values(ascending=False).head(10).index.tolist()
        df_filtered = df[df['Country'].isin(top_countries)]
    elif countries is not None and 'Country' in df.columns:
        df_filtered = df[df['Country'].isin(countries)]
    else:
        df_filtered = df

    if 'Country' in df.columns:
        df_grouped = df_filtered.groupby(['Year', 'Country'])[metric].sum().reset_index()
        fig = px.line(df_grouped, x='Year', y=metric, color='Country')
    else:
        df_grouped = df_filtered.groupby('Year')[metric].sum().reset_index()
        fig = px.line(df_grouped, x='Year', y=metric)

    fig.update_layout(**get_plot_layout(f"{metric} Over Time by Country" if 'Country' in df.columns else f"{metric} Over Time"))
    st.plotly_chart(fig, use_container_width=True)

def plot_comparisons(df, countries=None, metrics=None):
    if 'Country' not in df.columns:
        st.error("Required column not found: 'Country'")
        return

    if countries is None:
        if 'Production' in df.columns:
            countries = df.groupby('Country')['Production'].sum().sort_values(ascending=False).head(10).index.tolist()
        else:
            countries = df['Country'].unique().tolist()[:10]

    if metrics is None:
        potential_metrics = ['Production', 'Import Quantity', 'Export Quantity', 'Food', 'Feed', 'Losses']
        metrics = [m for m in potential_metrics if m in df.columns][:3]

    df_filtered = df[df['Country'].isin(countries)]
    df_agg = df_filtered.groupby('Country')[metrics].sum().reset_index()

    fig = px.bar(df_agg, x='Country', y=metrics, barmode='group')
    fig.update_layout(**get_plot_layout('Country Comparison by Key Metrics'))
    st.plotly_chart(fig, use_container_width=True)

def plot_food_balance(df, country):
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
    if not all(col in df.columns for col in ['Country', 'metric']):
        st.error(f"Required columns not found: 'Country' or '{metric}'")
        return

    if year is not None and 'Year' in df.columns:
        df_filtered = df[df['Year'] == year]
    elif 'Year' in df.columns:
        latest_year = df['Year'].max()
        df_filtered = df[df['Year'] == latest_year]
    else:
        df_filtered = df

    df_agg = df_filtered.groupby('Country')[metric].sum().reset_index()

    fig = px.choropleth(df_agg, locations='Country', locationmode='country names',
                        color=metric, hover_name='Country',
                        color_continuous_scale=px.colors.sequential.Viridis)
    fig.update_layout(**get_plot_layout(f'{metric} by Country', height=MAP_HEIGHT))
    st.plotly_chart(fig, use_container_width=True)

def plot_correlation_matrix(df, columns=None):
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