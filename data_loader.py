import pandas as pd
import numpy as np
import os

def load_and_preprocess_data(file_path="data/food_balance_sheet_europe.csv"):
    """
    Loads a CSV file and performs basic preprocessing.

    Args:
        file_path (str or Path): Path to the CSV file.

    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    # Load the data
    try:
        df = pd.read_csv(file_path)
        
        # Example preprocessing (adjust as needed)
        df.dropna(how='all', axis=1, inplace=True)  # Drop empty columns
        df.dropna(how='all', axis=0, inplace=True)  # Drop empty rows
        df.columns = df.columns.str.strip()         # Clean column names

        return df
    
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        print(f"Current working directory: {os.getcwd()}")
        print("Please make sure the data file exists at the specified path.")
        return None  # Return None instead of empty DataFrame for better error handling

def perform_eda(df):
    """
    Perform exploratory data analysis on a DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        
    Returns:
        dict: Dictionary containing EDA results with:
            - shape: DataFrame dimensions
            - columns: List of column names
            - missing_values: Count of missing values per column
            - data_types: Data types of each column
            - numeric_stats: Statistics for numeric columns
    """
    if df is None:
        raise ValueError("Cannot perform EDA on None DataFrame")
        
    return {
        'shape': df.shape,
        'columns': df.columns.tolist(),
        'missing_values': df.isnull().sum().to_dict(),
        'data_types': df.dtypes.astype(str).to_dict(),
        'numeric_stats': df.describe().to_dict() if df.select_dtypes(include=np.number).shape[1] > 0 else {}
    }