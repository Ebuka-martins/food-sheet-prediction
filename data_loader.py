import pandas as pd
import numpy as np
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_and_preprocess_data(file_path="data/input/FAOSTAT_data_10-23-2018.csv", encoding='utf-8', dtype=None):
    """
    Loads a CSV file and performs preprocessing for FAOSTAT data.

    Args:
        file_path (str): Path to the CSV file.
        encoding (str): Encoding type used to decode the file.
        dtype (dict): Dictionary specifying column data types.

    Returns:
        pd.DataFrame: Cleaned DataFrame or None if file is not found.
    """
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            logging.error(f"File not found at path: {file_path}")
            logging.error(f"Current working directory: {os.getcwd()}")
            return None

        # Load the dataset
        df = pd.read_csv(file_path, encoding=encoding, dtype=dtype)
        logging.info(f"Data loaded successfully from '{file_path}' with shape {df.shape}")

        # Preprocessing steps
        df.dropna(how='all', axis=1, inplace=True)  # Drop completely empty columns
        df.dropna(how='all', axis=0, inplace=True)  # Drop completely empty rows
        df.columns = df.columns.str.strip()         # Clean column names

        # FAOSTAT-specific preprocessing
        # Ensure critical columns exist
        required_columns = ['Country', 'Item', 'Element', 'Year', 'Value']
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            logging.error(f"Missing required columns: {missing_cols}")
            return None

        # Convert Year to integer and Value to numeric
        df['Year'] = pd.to_numeric(df['Year'], errors='coerce').astype('Int64')  # Handle missing years
        df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
        df = df.dropna(subset=['Year', 'Value'])  # Drop rows with missing Year or Value

        # Remove duplicates
        df = df.drop_duplicates()

        # Log unique values for debugging
        logging.info(f"Countries: {df['Country'].unique().tolist()}")
        logging.info(f"Items: {df['Item'].unique().tolist()}")
        logging.info(f"Elements: {df['Element'].unique().tolist()}")
        logging.info(f"Years: {df['Year'].unique().tolist()}")

        return df

    except FileNotFoundError:
        logging.error(f"File not found at path: {file_path}")
        logging.error(f"Current working directory: {os.getcwd()}")
        logging.error("Please ensure the data file exists at the specified location.")
        return None

    except Exception as e:
        logging.exception(f"An error occurred while loading the data: {e}")
        return None


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
    if df is None or df.empty:
        raise ValueError("Cannot perform EDA on None or empty DataFrame")
        
    return {
        'shape': df.shape,
        'columns': df.columns.tolist(),
        'missing_values': df.isnull().sum().to_dict(),
        'data_types': df.dtypes.astype(str).to_dict(),
        'numeric_stats': df.describe().to_dict() if df.select_dtypes(include=np.number).shape[1] > 0 else {}
    }