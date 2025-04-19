import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not available. Install with 'pip install xgboost' to use XGBoost models.")

def prepare_data_for_modeling(df, features, target, test_size=0.2, random_state=42):
    """
    Prepare data for modeling by splitting into train and test sets
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    features : list
        List of feature columns
    target : str
        Target column
    test_size : float
        Test set proportion
    random_state : int
        Random state for reproducibility
        
    Returns:
    --------
    tuple
        X_train, X_test, y_train, y_test, preprocessor
    """
    # Check if all features and target exist in the dataframe
    for col in features + [target]:
        if col not in df.columns:
            raise ValueError(f"Column {col} not found in the dataframe")
    
    # Extract features and target
    X = df[features].copy()
    y = df[target].copy()
    
    # Identify numeric and categorical features
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Create preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ] if categorical_features else [
            ('num', StandardScaler(), numeric_features)
        ],
        remainder='passthrough'
    )
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    return X_train, X_test, y_train, y_test, preprocessor

def train_model(df, features, target, model_type='Linear Regression'):
    """
    Train a machine learning model
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    features : list
        List of feature columns
    target : str
        Target column
    model_type : str
        Type of model to train
        
    Returns:
    --------
    tuple
        Trained model, performance metrics, and feature importance
    """
    try:
        # Prepare data
        X_train, X_test, y_train, y_test, preprocessor = prepare_data_for_modeling(df, features, target)
        
        # Define model based on model_type
        if model_type == 'Linear Regression':
            model = LinearRegression()
            param_grid = {}
        elif model_type == 'Ridge':
            model = Ridge()
            param_grid = {'model__alpha': [0.01, 0.1, 1.0, 10.0, 100.0]}
        elif model_type == 'Lasso':
            model = Lasso()
            param_grid = {'model__alpha': [0.01, 0.1, 1.0, 10.0, 100.0]}
        elif model_type == 'Random Forest':
            model = RandomForestRegressor(random_state=42)
            param_grid = {
                'model__n_estimators': [50, 100],
                'model__max_depth': [None, 10, 20],
                'model__min_samples_split': [2, 5]
            }
        elif model_type == 'XGBoost':
            if XGBOOST_AVAILABLE:
                model = xgb.XGBRegressor(random_state=42)
                param_grid = {
                    'model__n_estimators': [50, 100],
                    'model__max_depth': [3, 5, 7],
                    'model__learning_rate': [0.01, 0.1]
                }
            else:
                raise ImportError("XGBoost is not available. Please install with 'pip install xgboost'")
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Create pipeline
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        
        # Use grid search if param_grid is not empty
        if param_grid:
            grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='neg_mean_squared_error')
            grid_search.fit(X_train, y_train)
            best_pipeline = grid_search.best_estimator_
        else:
            best_pipeline = pipeline
            best_pipeline.fit(X_train, y_train)
        
        # Make predictions
        y_pred = best_pipeline.predict(X_test)
        
        # Calculate metrics
        metrics = {
            'r2': r2_score(y_test, y_pred),
            'mae': mean_absolute_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred))
        }
        
        # Get feature importance if available
        feature_importance = {}
        
        return best_pipeline, metrics, feature_importance
    
    except Exception as e:
        print(f"Error training model: {e}")
        return None, {'r2': 0, 'mae': 0, 'rmse': 0}, {}

def predict(model, X):
    """
    Make predictions using a trained model
    
    Parameters:
    -----------
    model : Pipeline
        Trained scikit-learn pipeline
    X : pd.DataFrame
        Input features
        
    Returns:
    --------
    np.ndarray
        Predictions
    """
    try:
        return model.predict(X)
    except Exception as e:
        print(f"Error making predictions: {e}")
        return None