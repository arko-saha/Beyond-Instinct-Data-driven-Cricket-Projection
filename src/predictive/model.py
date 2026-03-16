"""
Modeling and evaluation module for cricket batting prediction.
Handles model training, evaluation, and comparison.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import shap

def evaluate_model(model, X, y, name="Model"):
    """
    Evaluate a regression model using R2, MAE, and RMSE.
    
    Args:
        model: Trained model with predict method
        X: Feature matrix
        y: Target values
        name: Model name for printing
    
    Returns:
        dict: Dictionary with metrics and predictions
    """
    preds = model.predict(X)
    r2 = r2_score(y, preds)
    mae = mean_absolute_error(y, preds)
    rmse = np.sqrt(mean_squared_error(y, preds))
    print(f"[{name}] R2: {r2:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}")
    return {"r2": r2, "mae": mae, "rmse": rmse, "preds": preds}

def train_baseline_models(X_train, y_train):
    """
    Train baseline regression models.
    
    Args:
        X_train: Training features
        y_train: Training targets
    
    Returns:
        dict: Dictionary of trained models
    """
    models = {}
    
    # Linear Regression
    lr = LinearRegression().fit(X_train, y_train)
    models['Linear Regression'] = lr
    
    # Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train, y_train)
    models['Random Forest'] = rf
    
    return models

def train_xgboost_model(X_train, y_train, X_val=None, y_val=None):
    """
    Train XGBoost regressor with early stopping if validation data provided.
    
    Args:
        X_train: Training features
        y_train: Training targets
        X_val: Validation features (optional)
        y_val: Validation targets (optional)
    
    Returns:
        Trained XGBRegressor model
    """
    xgb_model = XGBRegressor(
        n_estimators=1000, 
        learning_rate=0.05, 
        verbosity=1, 
        random_state=42, 
        early_stopping_rounds=50
    )
    
    if X_val is not None and y_val is not None:
        xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
    else:
        xgb_model.fit(X_train, y_train)
    
    return xgb_model

def compare_models(results, save_path=None):
    """
    Compare model performance with bar plots.
    
    Args:
        results: Dictionary of model results from evaluate_model
        save_path: Path to save the plot (optional)
    """
    res_df = pd.DataFrame(results).T[['r2', 'mae', 'rmse']]
    res_df.plot(kind='bar', subplots=True, layout=(1,3), figsize=(15, 5), title="Validation Performance")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def explain_model_with_shap(model, X_val, save_path=None):
    """
    Generate SHAP explanations for the model.
    
    Args:
        model: Trained tree-based model
        X_val: Validation features
        save_path: Path to save SHAP plots (optional)
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_val)
    
    # Summary plot
    shap.summary_plot(shap_values, X_val, show=False)
    if save_path:
        plt.savefig(f"{save_path}/shap_summary.png")
    plt.show()
    
    return explainer, shap_values