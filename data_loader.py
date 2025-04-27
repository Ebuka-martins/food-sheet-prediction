import pandas as pd
import numpy as np
import os
import logging
from typing import Optional, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Country configuration
EUROPEAN_COUNTRIES = [
    ["276", "Germany", 82794],
    ["250", "France", 66836],
    ["380", "Italy", 60724],
    ["826", "United Kingdom", 65765],
    ["724", "Spain", 46704],
    ["616", "Poland", 38017],
    
]

def _disaggregate_european_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Internal function to split European aggregate data into individual countries.
    
    Args:
        df: DataFrame containing FAOSTAT data with possible 'Europe' records
        
    Returns:
        DataFrame with European data disaggregated to country level
    """
    try:
        if 'Europe' not in df['Country'].values:
            logging.info("No European aggregate data found - skipping disaggregation")
            return df

        logging.info("Disaggregating European data to country level...")
        
        # Prepare country weights
        countries_df = pd.DataFrame(
            EUROPEAN_COUNTRIES,
            columns=['Country Code', 'Country', 'Population']
        )
        countries_df['Weight'] = countries_df['Population'] / countries_df['Population'].sum()

        # Process European records
        europe_rows = df[df['Country'] == 'Europe']
        non_europe_rows = df[df['Country'] != 'Europe']
        
        new_rows = []
        for _, row in europe_rows.iterrows():
            element = row['Element']
            
            for _, country in countries_df.iterrows():
                new_row = row.copy()
                new_row['Country Code'] = country['Country Code']
                new_row['Country'] = country['Country']
                
                if element == "Total Population - Both sexes":
                    new_row['Value'] = country['Population']
                else:
                    new_row['Value'] = row['Value'] * country['Weight']
                
                new_rows.append(new_row)

        # Combine results
        result_df = pd.concat([non_europe_rows, pd.DataFrame(new_rows)])
        logging.info(f"Disaggregated {len(europe_rows)} European records into {len(new_rows)} country records")
        
        return result_df

    except Exception as e:
        logging.error(f"Error during European data disaggregation: {str(e)}")
        raise

def load_and_preprocess_data(file_path: str = "data/food_balance_sheet_europe.csv", 
                           encoding: str = 'utf-8', 
                           dtype: Optional[Dict] = None,
                           disaggregate_europe: bool = True) -> Optional[pd.DataFrame]:
    """
    Loads and preprocesses FAOSTAT data, with optional European data disaggregation.
    
    Args:
        file_path: Path to the CSV file
        encoding: File encoding
        dtype: Column data types
        disaggregate_europe: Whether to split European aggregates into countries
        
    Returns:
        Cleaned DataFrame or None if error occurs
    """
    try:
        # List of fallback files to try if the primary file is missing
        fallback_files = [
            "data/food_balance_sheet_europe.csv",
            "data/FAOSTAT_data_10-23-2018.csv",
            "data/food_balance_sheet_preprocessed.csv",
            "data/food_balance_stats.csv"
        ]
        
        # Check if the specified file exists
        selected_file = file_path
        if not os.path.exists(file_path):
            logging.error(f"File not found at path: {file_path}")
            logging.error(f"Current working directory: {os.getcwd()}")
            logging.info("Available files in data directory:")
            data_dir = os.path.dirname(file_path)
            if os.path.exists(data_dir):
                for f in os.listdir(data_dir):
                    logging.info(f" - {f}")
            
            # Try fallback files
            for fallback in fallback_files:
                if os.path.exists(fallback):
                    logging.info(f"Attempting to load fallback file: {fallback}")
                    selected_file = fallback
                    break
            else:
                logging.error("No valid CSV files found in data directory")
                return None
        else:
            logging.info(f"Found file: {file_path}")

        # Load data
        df = pd.read_csv(selected_file, encoding=encoding, dtype=dtype)
        logging.info(f"Data loaded successfully from '{selected_file}' with shape {df.shape}")

        # Log all columns
        logging.info(f"Dataset columns: {df.columns.tolist()}")

        # Basic cleaning
        df.dropna(how='all', axis=1, inplace=True)  
        df.dropna(how='all', axis=0, inplace=True)  
        df.columns = df.columns.str.strip()

        # Validate required columns
        required_columns = ['Country', 'Element', 'Year', 'Value']
        optional_columns = ['Item', 'Country Code', 'Item Code', 'Element Code', 'Year Code']
        missing_required = [col for col in required_columns if col not in df.columns]
        if missing_required:
            logging.warning(f"Missing required columns: {missing_required}")
            logging.info("Attempting to rename columns to match required format")
            rename_map = {
                'country': 'Country',
                'element': 'Element',
                'year': 'Year',
                'value': 'Value',
                'item': 'Item',
                'country_code': 'Country Code',
                'item_code': 'Item Code',
                'element_code': 'Element Code',
                'year_code': 'Year Code'
            }
            for old_col, new_col in rename_map.items():
                if old_col in df.columns and new_col not in df.columns:
                    df.rename(columns={old_col: new_col}, inplace=True)
                    logging.info(f"Renamed column '{old_col}' to '{new_col}'")
            # Re-check required columns
            missing_required = [col for col in required_columns if col not in df.columns]
            if missing_required:
                logging.error(f"Still missing required columns after renaming: {missing_required}")
                logging.info(f"Available columns after renaming: {df.columns.tolist()}")
                return None

        # Type conversion
        if 'Year' in df.columns:
            df['Year'] = pd.to_numeric(df['Year'], errors='coerce').astype('Int64')
        if 'Year Code' in df.columns:
            df['Year Code'] = pd.to_numeric(df['Year Code'], errors='coerce').astype('Int64')
        df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
        df = df.dropna(subset=['Year', 'Value'])

        # Deduplication
        df = df.drop_duplicates()

        # European data disaggregation
        if disaggregate_europe and 'Country' in df.columns:
            df = _disaggregate_european_data(df)

        # Log dataset summary
        logging.info(f"Processed dataset shape: {df.shape}")
        if 'Country' in df.columns:
            logging.info(f"Countries: {df['Country'].unique().tolist()[:10]}...")
        if 'Element' in df.columns:
            logging.info(f"Elements: {df['Element'].unique().tolist()[:10]}...")
        if 'Item' in df.columns:
            logging.info(f"Items: {df['Item'].unique().tolist()[:10]}...")
        if 'Year' in df.columns:
            logging.info(f"Years: {sorted(df['Year'].unique().tolist())}")
        if 'Year Code' in df.columns:
            logging.info(f"Year Codes: {sorted(df['Year Code'].unique().tolist())}")

        return df

    except Exception as e:
        logging.exception(f"An error occurred while loading the data from '{selected_file}': {e}")
        return None

def perform_eda(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Perform exploratory data analysis on the dataset.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary containing EDA results
    """
    if df is None or df.empty:
        raise ValueError("Cannot perform EDA on None or empty DataFrame")
        
    eda_results = {
        'shape': df.shape,
        'columns': df.columns.tolist(),
        'missing_values': df.isnull().sum().to_dict(),
        'data_types': df.dtypes.astype(str).to_dict(),
        'numeric_stats': df.describe().to_dict() if df.select_dtypes(include=np.number).shape[1] > 0 else {}
    }
    
    if 'Country' in df.columns:
        eda_results['unique_countries'] = df['Country'].nunique()
    if 'Element' in df.columns:
        eda_results['unique_elements'] = df['Element'].nunique()
    if 'Item' in df.columns:
        eda_results['unique_items'] = df['Item'].nunique()
    if 'Year' in df.columns:
        eda_results['unique_years'] = sorted(df['Year'].unique().tolist())
    
    return eda_results