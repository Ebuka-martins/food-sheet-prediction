import pandas as pd
import numpy as np
import io

def perform_eda(df):
    """
    Perform exploratory data analysis on the dataset.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe

    Returns:
    --------
    dict
        Dictionary containing EDA results
    """
    results = {
        'row_count': len(df),
        'column_count': len(df.columns),
        'missing_values': df.isna().sum().to_dict()
    }
    
    if 'Country' in df.columns:
        results['unique_countries'] = df['Country'].nunique()
        results['countries'] = sorted(df['Country'].unique().tolist())
    
    if 'Item' in df.columns:
        results['unique_items'] = df['Item'].nunique()
        results['items'] = sorted(df['Item'].unique().tolist())
    
    if 'Element' in df.columns:
        results['unique_elements'] = df['Element'].nunique()
        results['elements'] = sorted(df['Element'].unique().tolist())
    
    if 'Year' in df.columns:
        results['year_range'] = (int(df['Year'].min()), int(df['Year'].max())) if not df['Year'].isna().all() else (None, None)
    
    return results

def get_key_insights(df):
    """
    Generate key insights from the dataset, compatible with app.py.

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
    
    # Handle Production or Value column
    value_col = 'Production' if 'Production' in df.columns else 'Value' if 'Value' in df.columns else None
    
    if 'Country' in df.columns and value_col:
        top_producers = df.groupby('Country')[value_col].sum().sort_values(ascending=False)
        if not top_producers.empty:
            insights.append(f"Top producing country: {top_producers.index[0]} ({top_producers.iloc[0]:,.0f} units)")
    
    if 'Year' in df.columns and value_col:
        yearly_values = df.groupby('Year')[value_col].sum()
        if len(yearly_values) > 1:
            first_year = yearly_values.index.min()
            last_year = yearly_values.index.max()
            change = yearly_values.iloc[-1] - yearly_values.iloc[0]
            pct_change = (change / yearly_values.iloc[0]) * 100 if yearly_values.iloc[0] != 0 else float('inf')
            trend = "increased" if change > 0 else "decreased"
            insights.append(f"Total {value_col.lower()} {trend} by {abs(pct_change):.1f}% from {int(first_year)} to {int(last_year)}")
    
    if 'Item' in df.columns and value_col:
        top_items = df.groupby('Item')[value_col].sum().sort_values(ascending=False)
        if not top_items.empty:
            insights.append(f"Most produced item: {top_items.index[0]} ({top_items.iloc[0]:,.0f} units)")
    
    if 'Element' in df.columns and value_col:
        top_elements = df.groupby('Element')[value_col].sum().sort_values(ascending=False)
        if not top_elements.empty:
            insights.append(f"Top element: {top_elements.index[0]} ({top_elements.iloc[0]:,.0f} units)")
    
    if not insights:
        insights.append("No insights generated. Ensure the dataset contains columns like Country, Year, Item, Element, and Production/Value.")
    
    return insights

def calculate_element_year_statistics(df):
    """
    Calculate sum, mean, and std for numerical columns grouped by Element and Year.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe

    Returns:
    --------
    pd.DataFrame
        DataFrame with columns Element, Year, sum, mean, std
    """
    if not all(col in df.columns for col in ['Element', 'Year']):
        return pd.DataFrame(columns=['Element', 'Year', 'sum', 'mean', 'std'])
    
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    numerical_cols = [col for col in numerical_cols if col not in ['Year', 'Year Code', 'Country Code', 'Element Code', 'Item Code']]
    
    if not numerical_cols:
        return pd.DataFrame(columns=['Element', 'Year', 'sum', 'mean', 'std'])
    
    result = []
    grouped = df.groupby(['Element', 'Year'])
    
    for (element, year), group in grouped:
        values = group[numerical_cols].stack().dropna()
        total_sum = values.sum() if not values.empty else np.nan
        total_mean = values.mean() if not values.empty else np.nan
        total_std = values.std() if not values.empty else np.nan
        
        result.append({
            'Element': element,
            'Year': year,
            'sum': total_sum,
            'mean': total_mean,
            'std': total_std
        })
    
    result_df = pd.DataFrame(result)
    return result_df[['Element', 'Year', 'sum', 'mean', 'std']]

def get_summary_statistics(df):
    """
    Generate comprehensive summary statistics, including element-year stats.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe

    Returns:
    --------
    dict
        Dictionary containing describe, info, and element-year statistics
    """
    buffer = io.StringIO()
    df.info(buf=buffer)
    info_str = buffer.getvalue()
    
    summary = {
        'describe': df.describe(include='all').to_dict(),
        'info': info_str,
        'element_year_stats': calculate_element_year_statistics(df).to_dict(orient='records')
    }
    
    return summary

def prepare_country_comparison_data(df, countries):
    """
    Prepare data for country comparison chart by pivoting Element and Value.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe (filtered by countries)
    countries : list
        List of selected countries

    Returns:
    --------
    pd.DataFrame
        DataFrame with columns Country and metrics (e.g., Production, Import Quantity)
    """
    if not {'Country', 'Element', 'Value'}.issubset(df.columns) or not df['Country'].isin(countries).any():
        return pd.DataFrame(columns=['Country', 'Production', 'Import Quantity', 'Export Quantity'])
    
    df_filtered = df[df['Country'].isin(countries)][['Country', 'Element', 'Value']].copy()
    
    comparison_data = df_filtered.pivot_table(
        index='Country',
        columns='Element',
        values='Value',
        aggfunc='sum',
        fill_value=0
    ).reset_index()
    
    expected_metrics = ['Production', 'Import Quantity', 'Export Quantity', 'Food', 'Feed', 'Losses']

    for metric in expected_metrics:
        if metric not in comparison_data.columns:
            comparison_data[metric] = 0
    
    return comparison_data[['Country'] + [col for col in expected_metrics if col in comparison_data.columns]]