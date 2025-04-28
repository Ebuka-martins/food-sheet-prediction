import pandas as pd
import numpy as np
import os
import joblib
import logging
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logging.warning("XGBoost not available. Install with 'pip install xgboost' to use XGBoost models.")

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def save_model(model, filepath="models/best_model.joblib"):
    """Save trained model to disk"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)
    logging.info(f"Model saved to {filepath}")

def load_model(filepath="models/best_model.joblib"):
    """Load trained model from disk"""
    if os.path.exists(filepath):
        model = joblib.load(filepath)
        logging.info(f"Model loaded from {filepath}")
        return model
    logging.warning(f"Model file not found at {filepath}")
    return None

def prepare_data_for_modeling(df, features, target, test_size=0.3, random_state=42):
    """Prepare data for machine learning"""
    # Validate columns exist
    missing_cols = [col for col in features + [target] if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Columns not found in dataframe: {missing_cols}")
    
    X = df[features].copy()
    y = df[target].copy()

    # Check for NaN values
    if X.isna().any().any():
        logging.warning(f"Data contains {X.isna().sum().sum()} NaN values in features.")
    if y.isna().any():
        raise ValueError(f"Target variable '{target}' contains {y.isna().sum()} NaN values which must be handled.")

    # Log unique target values
    logging.info(f"Unique target values ({target}): {y.unique()}")

    # Identify feature types
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

    # Create preprocessing pipeline
    transformers = []
    if numeric_features:
        transformers.append(('num', StandardScaler(), numeric_features))
    if categorical_features:
        transformers.append(('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features))

    preprocessor = ColumnTransformer(
        transformers=transformers if transformers else [('passthrough', 'passthrough', features)],
        remainder='drop'
    )

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state
    )
    
    return X_train, X_test, y_train, y_test, preprocessor

def train_model(df, features, target, model_type='Linear Regression', progress_callback=None):
    """Train a machine learning model"""
    try:
        # Update progress
        if progress_callback:
            progress_callback(10)

        # Prepare data
        X_train, X_test, y_train, y_test, preprocessor = prepare_data_for_modeling(
            df, features, target
        )
        
        if progress_callback:
            progress_callback(20)

        # Model configuration
        model_config = {
            'Linear Regression': {
                'model': LinearRegression(),
                'params': {}
            },
            'Ridge': {
                'model': Ridge(),
                'params': {'model__alpha': [0.01, 0.1, 1.0, 10.0]}
            },
            'Lasso': {
                'model': Lasso(),
                'params': {'model__alpha': [0.01, 0.1, 1.0, 10.0]}
            },
            'Random Forest': {
                'model': RandomForestRegressor(random_state=42),
                'params': {
                    'model__n_estimators': [50, 100],
                    'model__max_depth': [None, 5, 10],
                    'model__min_samples_split': [2, 5]
                }
            },
            'XGBoost': {
                'model': xgb.XGBRegressor(random_state=42) if XGBOOST_AVAILABLE else None,
                'params': {
                    'model__n_estimators': [50, 100],
                    'model__max_depth': [3, 5],
                    'model__learning_rate': [0.01, 0.1]
                } if XGBOOST_AVAILABLE else {}
            }
        }

        # Validate model type
        if model_type not in model_config:
            raise ValueError(f"Unsupported model type: {model_type}")
        if model_type == 'XGBoost' and not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost is not available. Please install with 'pip install xgboost'")

        if progress_callback:
            progress_callback(30)

        # Create pipeline
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', model_config[model_type]['model'])
        ])

        # Train model
        param_grid = model_config[model_type]['params']
        if param_grid:
            scoring = 'neg_mean_squared_error'
            grid_search = GridSearchCV(
                pipeline, 
                param_grid, 
                cv=3, 
                scoring=scoring,
                n_jobs=-1
            )
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            logging.info(f"Best parameters: {grid_search.best_params_}")
        else:
            best_model = pipeline
            best_model.fit(X_train, y_train)

        if progress_callback:
            progress_callback(70)

        # Evaluate model
        y_pred = best_model.predict(X_test)
        metrics = {
            'r2': r2_score(y_test, y_pred),
            'mae': mean_absolute_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred))
        }
        logging.info(f"Model performance: {metrics}")
        logging.info(f"Unique predicted values: {np.unique(y_pred)}")

        if progress_callback:
            progress_callback(100)

        return best_model, metrics, {
            'X_test': X_test,
            'y_test': y_test,
            'y_pred': y_pred
        }

    except Exception as e:
        logging.error(f"Error in train_model: {str(e)}", exc_info=True)
        if progress_callback:
            progress_callback(0)
        return None, {'r2': 0, 'mae': 0, 'rmse': 0}, {}

def predict(model, X):
    """Make predictions using trained model"""
    try:
        if model is None:
            raise ValueError("Model not initialized")
        
        if isinstance(X, dict):
            X = pd.DataFrame([X])
        
        # Ensure consistent data types
        for col in X.columns:
            if X[col].dtype == object:
                X[col] = pd.to_numeric(X[col], errors='coerce')
        
        predictions = model.predict(X)
        if predictions.size == 0:
            raise ValueError("Empty predictions returned")
            
        return float(predictions[0]) if predictions.size == 1 else predictions
    
    except Exception as e:
        logging.error(f"Prediction error: {str(e)}", exc_info=True)
        raise ValueError(f"Prediction failed: {str(e)}")