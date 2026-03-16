"""
Model training module for cricket run prediction.
Handles training and hyperparameter tuning of ML models.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import tensorflow as tf
from tensorflow import keras
from imblearn.over_sampling import SMOTER
from sklearn.utils.class_weight import compute_sample_weight
import logging
import joblib
from pathlib import Path

logger = logging.getLogger(__name__)


class CricketModelTrainer:
    """Handles training and evaluation of cricket prediction models."""

    def __init__(self, models_dir: str = "models", random_state: int = 42):
        """
        Initialize the model trainer.

        Args:
            models_dir: Directory to save trained models
            random_state: Random state for reproducibility
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        self.random_state = random_state
        self.trained_models = {}
        self.best_params = {}

    def train_random_forest(self, X: pd.DataFrame, y: pd.Series,
                          tune_hyperparams: bool = True) -> RandomForestRegressor:
        """
        Train Random Forest model with optional hyperparameter tuning.

        Args:
            X: Feature matrix
            y: Target vector
            tune_hyperparams: Whether to perform hyperparameter tuning

        Returns:
            Trained Random Forest model
        """
        logger.info("Training Random Forest model")

        if tune_hyperparams:
            # Define hyperparameter grid
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2],
                'random_state': [self.random_state]
            }

            rf = RandomForestRegressor()
            grid_search = GridSearchCV(
                rf, param_grid, cv=3, scoring='neg_mean_squared_error',
                n_jobs=-1, verbose=1
            )
            grid_search.fit(X, y)

            self.best_params['random_forest'] = grid_search.best_params_
            model = grid_search.best_estimator_

            logger.info(f"Best RF parameters: {grid_search.best_params_}")

        else:
            # Use default parameters
            model = RandomForestRegressor(
                n_estimators=200,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=self.random_state,
                n_jobs=-1
            )
            model.fit(X, y)

        self.trained_models['random_forest'] = model
        self._save_model(model, 'random_forest.joblib')

        return model

    def train_decision_tree(self, X: pd.DataFrame, y: pd.Series,
                          tune_hyperparams: bool = True) -> DecisionTreeRegressor:
        """
        Train Decision Tree model with optional hyperparameter tuning.

        Args:
            X: Feature matrix
            y: Target vector
            tune_hyperparams: Whether to perform hyperparameter tuning

        Returns:
            Trained Decision Tree model
        """
        logger.info("Training Decision Tree model")

        if tune_hyperparams:
            param_grid = {
                'max_depth': [3, 4, 5, 6, 7],
                'min_samples_split': [2, 3, 4, 5, 10],
                'random_state': [self.random_state]
            }

            dt = DecisionTreeRegressor()
            grid_search = GridSearchCV(
                dt, param_grid, cv=3, scoring='neg_mean_squared_error',
                n_jobs=-1, verbose=1
            )
            grid_search.fit(X, y)

            self.best_params['decision_tree'] = grid_search.best_params_
            model = grid_search.best_estimator_

        else:
            model = DecisionTreeRegressor(
                max_depth=5,
                min_samples_split=5,
                random_state=self.random_state
            )
            model.fit(X, y)

        self.trained_models['decision_tree'] = model
        self._save_model(model, 'decision_tree.joblib')

        return model

    def train_xgboost(self, X: pd.DataFrame, y: pd.Series,
                     tune_hyperparams: bool = True) -> xgb.XGBRegressor:
        """
        Train XGBoost model with optional hyperparameter tuning.

        Args:
            X: Feature matrix
            y: Target vector
            tune_hyperparams: Whether to perform hyperparameter tuning

        Returns:
            Trained XGBoost model
        """
        logger.info("Training XGBoost model")

        if tune_hyperparams:
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.3],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0],
                'random_state': [self.random_state]
            }

            xgb_model = xgb.XGBRegressor(objective='reg:squarederror', eval_metric='rmse')
            grid_search = GridSearchCV(
                xgb_model, param_grid, cv=3, scoring='neg_mean_squared_error',
                n_jobs=-1, verbose=1
            )
            grid_search.fit(X, y)

            self.best_params['xgboost'] = grid_search.best_params_
            model = grid_search.best_estimator_

        else:
            model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.3,
                subsample=1.0,
                colsample_bytree=1.0,
                random_state=self.random_state,
                objective='reg:squarederror'
            )
            model.fit(X, y)

        self.trained_models['xgboost'] = model
        self._save_model(model, 'xgboost.joblib')

        return model

    def train_neural_network(self, X: pd.DataFrame, y: pd.Series,
                           architecture: str = 'advanced') -> keras.Model:
        """
        Train Neural Network model.

        Args:
            X: Feature matrix
            y: Target vector
            architecture: Network architecture ('simple', 'advanced', 'residual')

        Returns:
            Trained Keras model
        """
        logger.info(f"Training Neural Network ({architecture} architecture)")

        # Split data for validation
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state
        )

        # Scale features
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        # Create model based on architecture
        if architecture == 'simple':
            model = self._create_simple_nn(X_train.shape[1])
        elif architecture == 'advanced':
            model = self._create_advanced_nn(X_train.shape[1])
        elif architecture == 'residual':
            model = self._create_residual_nn(X_train.shape[1])
        else:
            raise ValueError(f"Unknown architecture: {architecture}")

        # Callbacks
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=10, restore_best_weights=True
        )
        lr_scheduler = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6
        )

        # Train model
        history = model.fit(
            X_train_scaled, y_train,
            epochs=100, batch_size=64,
            validation_data=(X_val_scaled, y_val),
            callbacks=[early_stopping, lr_scheduler],
            verbose=0
        )

        self.trained_models[f'neural_network_{architecture}'] = model
        self._save_model(model, f'neural_network_{architecture}.h5')
        self._save_model(scaler, f'scaler_{architecture}.joblib')

        logger.info(f"Neural Network training completed. Final val_loss: {history.history['val_loss'][-1]:.4f}")

        return model

    def train_weighted_random_forest(self, X: pd.DataFrame, y: pd.Series) -> RandomForestRegressor:
        """
        Train Random Forest with sample weights to handle imbalance.

        Args:
            X: Feature matrix
            y: Target vector

        Returns:
            Trained weighted Random Forest model
        """
        logger.info("Training Weighted Random Forest for imbalance handling")

        # Calculate sample weights (inverse frequency)
        runs_weights = 1 / y.value_counts()
        sample_weights = y.map(runs_weights).fillna(1.0)

        model = RandomForestRegressor(**self.best_params.get('random_forest',
            {'n_estimators': 200, 'max_depth': None, 'min_samples_split': 2,
             'min_samples_leaf': 1, 'random_state': self.random_state}))

        model.fit(X, y, sample_weight=sample_weights)

        self.trained_models['weighted_random_forest'] = model
        self._save_model(model, 'weighted_random_forest.joblib')

        return model

    def train_smoter_random_forest(self, X: pd.DataFrame, y: pd.Series) -> RandomForestRegressor:
        """
        Train Random Forest with SMOTE for regression.

        Args:
            X: Feature matrix
            y: Target vector

        Returns:
            Trained SMOTER Random Forest model
        """
        logger.info("Training Random Forest with SMOTER")

        # Bin the target for oversampling
        y_binned = pd.cut(y, bins=[-1, 0, 1, 2, 6], labels=[0, 1, 2, 3])

        # Apply SMOTER
        smoter = SMOTER(random_state=self.random_state, k_neighbors=5)
        X_resampled, y_binned_resampled = smoter.fit_resample(X, y_binned)

        # Convert back to original scale
        y_resampled = y_binned_resampled.astype(int)

        model = RandomForestRegressor(**self.best_params.get('random_forest',
            {'n_estimators': 200, 'max_depth': None, 'min_samples_split': 2,
             'min_samples_leaf': 1, 'random_state': self.random_state}))

        model.fit(X_resampled, y_resampled)

        self.trained_models['smoter_random_forest'] = model
        self._save_model(model, 'smoter_random_forest.joblib')

        return model

    def _create_simple_nn(self, input_dim: int) -> keras.Model:
        """Create simple neural network."""
        model = keras.Sequential([
            keras.layers.Dense(64, activation='relu', input_shape=(input_dim,)),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model

    def _create_advanced_nn(self, input_dim: int) -> keras.Model:
        """Create advanced neural network with regularization."""
        from tensorflow.keras import regularizers

        model = keras.Sequential([
            keras.layers.Dense(128, activation='relu', input_shape=(input_dim,),
                              kernel_regularizer=regularizers.l2(0.001)),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.1),
            keras.layers.Dense(16, activation='relu'),
            keras.layers.Dense(1)
        ])
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                      loss='mse', metrics=['mae', 'mse'])
        return model

    def _create_residual_nn(self, input_dim: int) -> keras.Model:
        """Create residual neural network."""
        inputs = keras.Input(shape=(input_dim,))

        # First block
        x = keras.layers.Dense(64, activation='relu')(inputs)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dropout(0.1)(x)

        # Residual connection
        residual = x
        x = keras.layers.Dense(64, activation='relu')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Add()([x, residual])
        x = keras.layers.Dropout(0.1)(x)

        # Second block
        residual = keras.layers.Dense(32)(x)
        x = keras.layers.Dense(32, activation='relu')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Add()([x, residual])
        x = keras.layers.Dropout(0.1)(x)

        outputs = keras.layers.Dense(1)(x)

        model = keras.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                      loss='mse', metrics=['mae'])
        return model

    def _save_model(self, model, filename: str):
        """Save trained model to disk."""
        model_path = self.models_dir / filename
        if hasattr(model, 'save'):  # Keras model
            model.save(model_path)
        else:  # Sklearn model
            joblib.dump(model, model_path)
        logger.info(f"Model saved to {model_path}")

    def load_model(self, model_name: str, model_type: str = 'sklearn'):
        """Load trained model from disk."""
        if model_type == 'sklearn':
            model_path = self.models_dir / f"{model_name}.joblib"
            model = joblib.load(model_path)
        else:  # Keras model
            model_path = self.models_dir / f"{model_name}.h5"
            model = keras.models.load_model(model_path)

        logger.info(f"Model loaded from {model_path}")
        return model

    def get_training_summary(self) -> Dict:
        """Get summary of trained models."""
        summary = {
            'trained_models': list(self.trained_models.keys()),
            'best_parameters': self.best_params,
            'models_directory': str(self.models_dir)
        }
        return summary