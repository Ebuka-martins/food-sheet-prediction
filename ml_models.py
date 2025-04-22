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
    print("XGBoost not available. Install with 'pip install xgboost' to use XGBoost models.")

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def save_model(model, filepath="models/best_model.joblib"):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)
    logging.info(f"Model saved to {filepath}")


def load_model(filepath="models/best_model.joblib"):
    if os.path.exists(filepath):
        logging.info(f"Model loaded from {filepath}")
        return joblib.load(filepath)
    else:
        logging.warning(f"Model file not found at {filepath}")
        return None


def prepare_data_for_modeling(df, features, target, test_size=0.2, random_state=42):
    for col in features + [target]:
        if col not in df.columns:
            raise ValueError(f"Column {col} not found in the dataframe")

    X = df[features].copy()
    y = df[target].copy()

    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ] if categorical_features else [
            ('num', StandardScaler(), numeric_features)
        ],
        remainder='passthrough'
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test, preprocessor


def train_model(df, features, target, model_type='Linear Regression', progress_callback=None):
    try:
        if progress_callback:
            progress_callback(10)

        X_train, X_test, y_train, y_test, preprocessor = prepare_data_for_modeling(df, features, target)

        if progress_callback:
            progress_callback(20)

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

        if progress_callback:
            progress_callback(30)

        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', model)
        ])

        if param_grid:
            grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='neg_mean_squared_error')
            grid_search.fit(X_train, y_train)
            best_pipeline = grid_search.best_estimator_
            logging.info(f"Best parameters for {model_type}: {grid_search.best_params_}")
        else:
            best_pipeline = pipeline
            best_pipeline.fit(X_train, y_train)

        if progress_callback:
            progress_callback(70)

        y_pred = best_pipeline.predict(X_test)

        metrics = {
            'r2': r2_score(y_test, y_pred),
            'mae': mean_absolute_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred))
        }

        logging.info(f"{model_type} model performance: R2={metrics['r2']:.3f}, MAE={metrics['mae']:.3f}, RMSE={metrics['rmse']:.3f}")

        if progress_callback:
            progress_callback(100)

        return best_pipeline, metrics, {}

    except Exception as e:
        logging.error(f"Error training model: {e}")
        if progress_callback:
            progress_callback(0)
        return None, {'r2': 0, 'mae': 0, 'rmse': 0}, {}


def predict(model, X):
    try:
        return model.predict(X)
    except Exception as e:
        logging.error(f"Error making predictions: {e}")
        return None
