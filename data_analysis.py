import pandas as pd
import numpy as np
import io

def perform_eda(df):
    """
    Perform exploratory data analysis on the dataset
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
        
    Returns:
    --------
    dict
        Dictionary containing EDA results
    """
    results = {}
    
    results['row_count'] = len(df)
    results['column_count'] = len(df.columns)
    results['missing_values'] = df.isna().sum().to_dict()
    
    if 'Country' in df.columns:
        results['unique_countries'] = df['Country'].nunique()
        results['countries'] = sorted(df['Country'].unique().tolist())
    
    if 'Item' in df.columns:
        results['unique_items'] = df['Item'].nunique()
        results['items'] = sorted(df['Item'].unique().tolist())
    
    if 'Year' in df.columns:
        results['year_range'] = (df['Year'].min(), df['Year'].max())
    
    return results


def get_key_insights(df):
    """
    Generate key insights from the dataset
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
        
    Returns:
    --------
    list
        List of insights
    """
    insights = []
    
    if all(col in df.columns for col in ['Country', 'Production']):
        top_producers = df.groupby('Country')['Production'].sum().sort_values(ascending=False)
        if not top_producers.empty:
            insights.append(f"Top producing country is {top_producers.index[0]} with {top_producers.iloc[0]:,.0f} units")
    
    if all(col in df.columns for col in ['Year', 'Production']):
        yearly_production = df.groupby('Year')['Production'].sum()
        if len(yearly_production) > 1:
            first_year = yearly_production.index.min()
            last_year = yearly_production.index.max()
            change = yearly_production.iloc[-1] - yearly_production.iloc[0]
            pct_change = (change / yearly_production.iloc[0]) * 100 if yearly_production.iloc[0] != 0 else float('inf')
            
            if pct_change > 0:
                insights.append(f"Production increased by {pct_change:.1f}% from {first_year} to {last_year}")
            else:
                insights.append(f"Production decreased by {abs(pct_change):.1f}% from {first_year} to {last_year}")
    
    if all(col in df.columns for col in ['Country', 'Import Quantity', 'Export Quantity']):
        df['Net_Trade'] = df['Export Quantity'] - df['Import Quantity']
        net_trade = df.groupby('Country')['Net_Trade'].sum().sort_values()
        
        if not net_trade.empty and len(net_trade) >= 2:
            insights.append(f"Largest net importer: {net_trade.index[0]}")
            insights.append(f"Largest net exporter: {net_trade.index[-1]}")
    
    if 'Item' in df.columns and 'Production' in df.columns:
        top_items = df.groupby('Item')['Production'].sum().sort_values(ascending=False)
        if not top_items.empty:
            insights.append(f"Most produced item: {top_items.index[0]} ({top_items.iloc[0]:,.0f} units)")
    
    if not insights:
        insights.append("Analyze the data further to discover more specific patterns")
    
    return insights


def calculate_food_security_metrics(df):
    """
    Calculate food security metrics
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with food security metrics
    """
    if not all(col in df.columns for col in ['Country', 'Production', 'Import Quantity', 'Export Quantity']):
        return pd.DataFrame()
    
    result_df = df.copy()
    
    if 'Food' in result_df.columns:
        result_df['Food_Availability'] = result_df['Production'] + result_df['Import Quantity'] - result_df['Export Quantity']
        result_df['Self_Sufficiency_Ratio'] = result_df['Production'] / result_df['Food']
        result_df['Import_Dependency_Ratio'] = result_df['Import Quantity'] / result_df['Food']
        
        for col in ['Self_Sufficiency_Ratio', 'Import_Dependency_Ratio']:
            result_df[col] = result_df[col].replace([np.inf, -np.inf], np.nan)
    
    if 'Country' in result_df.columns:
        metrics = ['Food_Availability', 'Self_Sufficiency_Ratio', 'Import_Dependency_Ratio']
        available_metrics = [m for m in metrics if m in result_df.columns]
        
        if available_metrics:
            country_metrics = result_df.groupby('Country')[available_metrics].mean().reset_index()
            return country_metrics
    
    return pd.DataFrame()


def get_summary_statistics(df):
    """
    Generate summary statistics including df.describe() and df.info()
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
        
    Returns:
    --------
    dict
        Dictionary with 'describe' and 'info' summaries
    """
    buffer = io.StringIO()
    df.info(buf=buffer)
    info_str = buffer.getvalue()
    
    return {
        'describe': df.describe(include='all').to_dict(),
        'info': info_str
    }