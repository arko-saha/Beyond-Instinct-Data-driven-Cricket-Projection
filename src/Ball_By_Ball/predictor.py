"""
Prediction module for cricket run prediction.
Handles real-time predictions and batch processing.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any
import joblib
import tensorflow as tf
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class RunPredictor:
    """Handles predictions using trained cricket models."""

    def __init__(self, models_dir: str = "models", model_name: str = "random_forest"):
        """
        Initialize the predictor.

        Args:
            models_dir: Directory containing trained models
            model_name: Default model to use for predictions
        """
        self.models_dir = Path(models_dir)
        self.model_name = model_name
        self.model = None
        self.scaler = None
        self.feature_engineer = None
        self.is_keras_model = False

        # Load the default model
        self.load_model(model_name)

    def load_model(self, model_name: str):
        """
        Load a trained model.

        Args:
            model_name: Name of the model to load
        """
        model_path = self.models_dir / f"{model_name}.joblib"

        if model_path.exists():
            # Sklearn model
            self.model = joblib.load(model_path)
            self.is_keras_model = False
            logger.info(f"Loaded sklearn model: {model_name}")

            # Try to load scaler if it exists
            scaler_path = self.models_dir / f"scaler_{model_name.split('_')[-1]}.joblib"
            if scaler_path.exists():
                self.scaler = joblib.load(scaler_path)
                logger.info(f"Loaded scaler for {model_name}")

        else:
            # Try Keras model
            keras_path = self.models_dir / f"{model_name}.h5"
            if keras_path.exists():
                self.model = tf.keras.models.load_model(keras_path)
                self.is_keras_model = True
                logger.info(f"Loaded Keras model: {model_name}")

                # Try to load scaler
                scaler_path = self.models_dir / f"scaler_{model_name.split('_')[-1]}.joblib"
                if scaler_path.exists():
                    self.scaler = joblib.load(scaler_path)
                    logger.info(f"Loaded scaler for {model_name}")
            else:
                raise FileNotFoundError(f"Model {model_name} not found in {self.models_dir}")

        self.model_name = model_name

    def predict_single_ball(self, ball_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict runs for a single ball.

        Args:
            ball_data: Dictionary containing ball features

        Returns:
            Dictionary with prediction results
        """
        # Convert to DataFrame
        df = pd.DataFrame([ball_data])

        # Preprocess features
        df_processed = self._preprocess_ball_data(df)

        # Make prediction
        prediction = self._predict(df_processed)

        # Round to nearest integer (cricket runs are discrete)
        predicted_runs = round(float(prediction[0]))

        # Calculate confidence (based on model type)
        confidence = self._calculate_confidence(df_processed, prediction)

        result = {
            'predicted_runs': predicted_runs,
            'raw_prediction': float(prediction[0]),
            'confidence_score': confidence,
            'model_used': self.model_name,
            'prediction_range': self._get_prediction_range(predicted_runs)
        }

        logger.info(f"Predicted {predicted_runs} runs (confidence: {confidence:.2f})")
        return result

    def predict_batch(self, ball_data_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Predict runs for multiple balls.

        Args:
            ball_data_list: List of dictionaries containing ball features

        Returns:
            List of prediction results
        """
        logger.info(f"Predicting runs for {len(ball_data_list)} balls")

        # Convert to DataFrame
        df = pd.DataFrame(ball_data_list)

        # Preprocess features
        df_processed = self._preprocess_ball_data(df)

        # Make predictions
        predictions = self._predict(df_processed)

        # Process results
        results = []
        for i, pred in enumerate(predictions):
            predicted_runs = round(float(pred))
            confidence = self._calculate_confidence(df_processed.iloc[i:i+1], [pred])

            result = {
                'ball_index': i,
                'predicted_runs': predicted_runs,
                'raw_prediction': float(pred),
                'confidence_score': confidence,
                'model_used': self.model_name,
                'prediction_range': self._get_prediction_range(predicted_runs)
            }
            results.append(result)

        return results

    def predict_match(self, match_data: pd.DataFrame) -> pd.DataFrame:
        """
        Predict runs for an entire match.

        Args:
            match_data: DataFrame containing all balls in a match

        Returns:
            DataFrame with predictions added
        """
        logger.info(f"Predicting runs for match with {len(match_data)} balls")

        # Preprocess all balls
        df_processed = self._preprocess_ball_data(match_data.copy())

        # Make predictions
        predictions = self._predict(df_processed)

        # Add predictions to original data
        result_df = match_data.copy()
        result_df['predicted_runs'] = [round(float(pred)) for pred in predictions]
        result_df['raw_prediction'] = [float(pred) for pred in predictions]

        # Calculate confidence for each prediction
        confidences = []
        for i, pred in enumerate(predictions):
            conf = self._calculate_confidence(df_processed.iloc[i:i+1], [pred])
            confidences.append(conf)

        result_df['confidence_score'] = confidences

        # Calculate match-level statistics
        total_actual = result_df.get('runs_off_bat', 0).sum() if 'runs_off_bat' in result_df.columns else 0
        total_predicted = result_df['predicted_runs'].sum()
        match_accuracy = np.mean(result_df['predicted_runs'] == result_df.get('runs_off_bat', result_df['predicted_runs']))

        logger.info(f"Match prediction complete. Total predicted: {total_predicted}, Accuracy: {match_accuracy:.2f}")

        return result_df

    def _preprocess_ball_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess ball data for prediction.

        Args:
            df: Raw ball data

        Returns:
            Preprocessed DataFrame
        """
        # This is a simplified version - in production, you'd use the full
        # preprocessing pipeline from the CricketDataPreprocessor and CricketFeatureEngineer

        df_processed = df.copy()

        # Handle missing values (simple imputation)
        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df_processed[col].isnull().any():
                df_processed[col].fillna(df_processed[col].median(), inplace=True)

        # Encode categorical variables (simplified - assumes known categories)
        categorical_cols = df_processed.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            if df_processed[col].dtype.name == 'category':
                df_processed[col] = df_processed[col].cat.codes
            else:
                # Simple label encoding for new data
                unique_vals = df_processed[col].unique()
                mapping = {val: i for i, val in enumerate(unique_vals)}
                df_processed[col] = df_processed[col].map(mapping)

        # Ensure all values are numeric
        df_processed = df_processed.apply(pd.to_numeric, errors='coerce').fillna(0)

        return df_processed

    def _predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the loaded model.

        Args:
            X: Preprocessed feature matrix

        Returns:
            Array of predictions
        """
        if self.is_keras_model:
            # Keras model
            if self.scaler:
                X_scaled = self.scaler.transform(X)
            else:
                X_scaled = X.values
            predictions = self.model.predict(X_scaled, verbose=0).flatten()
        else:
            # Sklearn model
            predictions = self.model.predict(X.values)

        return predictions

    def _calculate_confidence(self, X: pd.DataFrame, prediction: List[float]) -> float:
        """
        Calculate prediction confidence score.

        Args:
            X: Feature matrix for the prediction
            prediction: Model prediction

        Returns:
            Confidence score between 0 and 1
        """
        if hasattr(self.model, 'predict_proba'):
            # For models with probability estimates
            try:
                proba = self.model.predict_proba(X.values)
                confidence = np.max(proba, axis=1)[0]
            except:
                confidence = 0.8  # Default confidence
        elif hasattr(self.model, 'estimators_'):
            # For ensemble models, use prediction variance
            if self.is_keras_model:
                confidence = 0.85  # Default for neural networks
            else:
                # Use standard deviation of tree predictions
                tree_predictions = np.array([tree.predict(X.values) for tree in self.model.estimators_])
                std_dev = np.std(tree_predictions, axis=0)[0]
                # Convert std dev to confidence (lower std = higher confidence)
                confidence = max(0.1, min(1.0, 1.0 / (1.0 + std_dev)))
        else:
            # Default confidence for other models
            confidence = 0.8

        return confidence

    def _get_prediction_range(self, predicted_runs: int) -> str:
        """
        Get human-readable prediction range.

        Args:
            predicted_runs: Predicted number of runs

        Returns:
            String describing the prediction range
        """
        if predicted_runs == 0:
            return "No runs expected"
        elif predicted_runs == 1:
            return "Single expected"
        elif predicted_runs == 2:
            return "Double expected"
        elif predicted_runs == 3:
            return "Triple expected"
        elif predicted_runs == 4:
            return "Boundary expected"
        elif predicted_runs == 6:
            return "Six expected"
        else:
            return f"{predicted_runs} runs expected"

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the currently loaded model."""
        info = {
            'model_name': self.model_name,
            'model_type': 'Keras' if self.is_keras_model else 'Sklearn',
            'has_scaler': self.scaler is not None,
            'models_dir': str(self.models_dir)
        }

        if hasattr(self.model, 'get_params'):
            info['model_params'] = self.model.get_params()

        return info

    def list_available_models(self) -> List[str]:
        """List all available trained models."""
        model_files = list(self.models_dir.glob("*.joblib")) + list(self.models_dir.glob("*.h5"))
        model_names = []

        for model_file in model_files:
            if model_file.suffix == '.joblib':
                model_names.append(model_file.stem)
            elif model_file.suffix == '.h5':
                model_names.append(model_file.stem)

        return sorted(list(set(model_names)))

    def validate_ball_data(self, ball_data: Dict[str, Any]) -> List[str]:
        """
        Validate ball data for required features.

        Args:
            ball_data: Dictionary containing ball features

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        # Check for critical features
        critical_features = ['batting_team', 'bowling_team', 'striker', 'bowler',
                           'completed_over', 'ball_no', 'wickets_remaining', 'balls_remaining']

        for feature in critical_features:
            if feature not in ball_data:
                errors.append(f"Missing critical feature: {feature}")

        # Check for numeric features
        numeric_features = ['wickets_remaining', 'balls_remaining', 'CRR', 'RRR']
        for feature in numeric_features:
            if feature in ball_data:
                try:
                    float(ball_data[feature])
                except (ValueError, TypeError):
                    errors.append(f"Invalid numeric value for {feature}: {ball_data[feature]}")

        return errors