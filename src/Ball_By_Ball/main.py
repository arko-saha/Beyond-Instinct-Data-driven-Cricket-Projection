#!/usr/bin/env python3
"""
Cricket Run Prediction - Main CLI Application
Production-grade command-line interface for ball-by-ball cricket run prediction.
"""

import argparse
import sys
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Lazy imports to avoid loading heavy dependencies at startup
def get_data_loader():
    from .data_loader import CricketDataLoader
    return CricketDataLoader

def get_preprocessor():
    from .preprocessor import CricketDataPreprocessor
    return CricketDataPreprocessor

def get_feature_engineer():
    from .feature_engineering import CricketFeatureEngineer
    return CricketFeatureEngineer

def get_model_trainer():
    from .model_trainer import CricketModelTrainer
    return CricketModelTrainer

def get_evaluator():
    from .evaluator import ModelEvaluator
    return ModelEvaluator

def get_predictor():
    from .predictor import RunPredictor
    return RunPredictor

# Configure logging
# Create logs directory if it doesn't exist
logs_dir = Path(__file__).parent / 'logs'
logs_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(logs_dir / 'cricket_prediction.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


class CricketPredictionCLI:
    """Command-line interface for cricket run prediction system."""

    def __init__(self):
        self.data_loader = None
        self.preprocessor = None
        self.feature_engineer = None
        self.model_trainer = None
        self.evaluator = None
        self.predictor = None

    def setup_components(self):
        """Initialize all system components."""
        logger.info("Initializing cricket prediction system components")

        self.data_loader = get_data_loader()()
        self.preprocessor = get_preprocessor()()
        self.feature_engineer = get_feature_engineer()()
        self.model_trainer = get_model_trainer()()
        self.evaluator = get_evaluator()()
        self.predictor = get_predictor()()

        logger.info("All components initialized successfully")

    def train_models(self, data_file: str, tune_hyperparams: bool = True):
        """Train all models on the provided dataset."""
        logger.info(f"Starting model training with data: {data_file}")

        # Load and preprocess data
        df = self.data_loader.load_historical_data(data_file)
        df_processed = self.preprocessor.preprocess_data(df)
        df_featured = self.feature_engineer.create_advanced_features(df_processed)
        df_encoded = self.feature_engineer.encode_categorical_features(df_featured)

        # Select features
        X, selected_features = self.feature_engineer.select_features(
            df_encoded, target_column='runs_off_bat', n_features=20
        )
        y = df_encoded['runs_off_bat']

        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42
        )

        logger.info(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples")

        # Train models
        results = {}

        # Random Forest
        rf_model = self.model_trainer.train_random_forest(
            X_train, y_train, tune_hyperparams=tune_hyperparams
        )
        rf_pred = rf_model.predict(X_test)
        results['Random Forest'] = self.evaluator.evaluate_model(y_test, rf_pred, 'Random Forest')

        # Decision Tree
        dt_model = self.model_trainer.train_decision_tree(
            X_train, y_train, tune_hyperparams=tune_hyperparams
        )
        dt_pred = dt_model.predict(X_test)
        results['Decision Tree'] = self.evaluator.evaluate_model(y_test, dt_pred, 'Decision Tree')

        # XGBoost
        xgb_model = self.model_trainer.train_xgboost(
            X_train, y_train, tune_hyperparams=tune_hyperparams
        )
        xgb_pred = xgb_model.predict(X_test)
        results['XGBoost'] = self.evaluator.evaluate_model(y_test, xgb_pred, 'XGBoost')

        # Neural Networks
        nn_model = self.model_trainer.train_neural_network(X_train, y_train, architecture='advanced')
        nn_pred = nn_model.predict(X_test).flatten()
        results['Neural Network'] = self.evaluator.evaluate_model(y_test, nn_pred, 'Neural Network')

        # Create comparison report
        comparison_df = self.evaluator.create_model_comparison_report(results)

        # Print results
        print("\n" + "="*60)
        print("MODEL TRAINING RESULTS")
        print("="*60)
        print(comparison_df.to_string(index=False, float_format='%.4f'))

        # Identify best model
        best_model = comparison_df.loc[comparison_df['r2'].idxmax(), 'Model']
        print(f"\n🏆 Best Model: {best_model} (R² = {comparison_df['r2'].max():.4f})")

        logger.info("Model training completed successfully")

    def predict_ball(self, ball_data: Dict[str, Any], model_name: str = "random_forest"):
        """Predict runs for a single ball."""
        # Load specified model
        self.predictor.load_model(model_name)

        # Validate input
        errors = self.predictor.validate_ball_data(ball_data)
        if errors:
            print("Validation errors:")
            for error in errors:
                print(f"  - {error}")
            return None

        # Make prediction
        result = self.predictor.predict_single_ball(ball_data)

        print("\n" + "="*40)
        print("BALL PREDICTION RESULT")
        print("="*40)
        print(f"Predicted Runs: {result['predicted_runs']}")
        print(f"Raw Prediction: {result['raw_prediction']:.3f}")
        print(f"Confidence: {result['confidence_score']:.2f}")
        print(f"Model Used: {result['model_used']}")
        print(f"Interpretation: {result['prediction_range']}")

        return result

    def predict_file(self, input_file: str, output_file: str, model_name: str = "random_forest"):
        """Predict runs for balls from a CSV file."""
        logger.info(f"Predicting runs from file: {input_file}")

        # Load data
        df = pd.read_csv(input_file)
        logger.info(f"Loaded {len(df)} balls for prediction")

        # Load model
        self.predictor.load_model(model_name)

        # Make predictions
        results_df = self.predictor.predict_match(df)

        # Save results
        results_df.to_csv(output_file, index=False)
        logger.info(f"Predictions saved to: {output_file}")

        # Print summary
        total_predicted = results_df['predicted_runs'].sum()
        avg_confidence = results_df['confidence_score'].mean()

        print(f"\nPredictions completed for {len(results_df)} balls")
        print(f"Total predicted runs: {total_predicted}")
        print(f"Average confidence: {avg_confidence:.2f}")
        print(f"Results saved to: {output_file}")

    def evaluate_models(self, data_file: str):
        """Evaluate all trained models."""
        logger.info("Starting model evaluation")

        # Load and preprocess data
        df = self.data_loader.load_historical_data(data_file)
        df_processed = self.preprocessor.preprocess_data(df)
        df_featured = self.feature_engineer.create_advanced_features(df_processed)
        df_encoded = self.feature_engineer.encode_categorical_features(df_featured)

        X, _ = self.feature_engineer.select_features(
            df_encoded, target_column='runs_off_bat', n_features=20
        )
        y = df_encoded['runs_off_bat']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42
        )

        # Evaluate each model
        available_models = self.predictor.list_available_models()
        results = {}

        for model_name in available_models:
            try:
                self.predictor.load_model(model_name)
                if model_name in ['neural_network_advanced', 'neural_network_residual']:
                    # Keras models need scaled data
                    scaler_path = Path("models") / f"scaler_{model_name.split('_')[-1]}.joblib"
                    if scaler_path.exists():
                        import joblib
                        scaler = joblib.load(scaler_path)
                        X_test_scaled = scaler.transform(X_test)
                        pred = self.predictor.model.predict(X_test_scaled, verbose=0).flatten()
                    else:
                        continue
                else:
                    pred = self.predictor.model.predict(X_test.values)

                results[model_name] = self.evaluator.evaluate_model(y_test, pred, model_name)

            except Exception as e:
                logger.error(f"Error evaluating {model_name}: {e}")
                continue

        if results:
            comparison_df = self.evaluator.create_model_comparison_report(results)
            print("\n" + "="*60)
            print("MODEL EVALUATION RESULTS")
            print("="*60)
            print(comparison_df.to_string(index=False, float_format='%.4f'))
        else:
            print("No models available for evaluation")

    def show_model_info(self):
        """Show information about available models."""
        available_models = self.predictor.list_available_models()

        print("\n" + "="*40)
        print("AVAILABLE MODELS")
        print("="*40)

        if not available_models:
            print("No trained models found in models/ directory")
            return

        for model in available_models:
            print(f"  - {model}")

        print(f"\nTotal models: {len(available_models)}")

        # Show current model info
        if self.predictor:
            current_info = self.predictor.get_model_info()
            print(f"\nCurrently loaded model: {current_info['model_name']}")
            print(f"Model type: {current_info['model_type']}")


def create_argument_parser():
    """Create command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Cricket Run Prediction System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train models
  python main.py train data/historical_data.csv

  # Predict single ball
  python main.py predict --ball-data '{"batting_team": "Australia", "wickets_remaining": 7, ...}'

  # Predict from file
  python main.py predict-file input_balls.csv predictions.csv

  # Evaluate models
  python main.py evaluate data/historical_data.csv

  # Show available models
  python main.py models
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Train command
    train_parser = subparsers.add_parser('train', help='Train models')
    train_parser.add_argument('data_file', help='Path to training data CSV file')
    train_parser.add_argument('--no-tune', action='store_true',
                             help='Skip hyperparameter tuning')

    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Predict runs for single ball')
    predict_parser.add_argument('--ball-data', required=True,
                               help='JSON string with ball data')
    predict_parser.add_argument('--model', default='random_forest',
                               help='Model to use for prediction')

    # Predict file command
    predict_file_parser = subparsers.add_parser('predict-file', help='Predict runs from CSV file')
    predict_file_parser.add_argument('input_file', help='Input CSV file with ball data')
    predict_file_parser.add_argument('output_file', help='Output CSV file for predictions')
    predict_file_parser.add_argument('--model', default='random_forest',
                                    help='Model to use for prediction')

    # Evaluate command
    evaluate_parser = subparsers.add_parser('evaluate', help='Evaluate trained models')
    evaluate_parser.add_argument('data_file', help='Path to test data CSV file')

    # Models command
    models_parser = subparsers.add_parser('models', help='Show available models')

    return parser


def main():
    """Main CLI entry point."""
    parser = create_argument_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Initialize CLI
    cli = CricketPredictionCLI()
    cli.setup_components()

    try:
        if args.command == 'train':
            tune = not args.no_tune
            cli.train_models(args.data_file, tune_hyperparams=tune)

        elif args.command == 'predict':
            ball_data = json.loads(args.ball_data)
            cli.predict_ball(ball_data, model_name=args.model)

        elif args.command == 'predict-file':
            cli.predict_file(args.input_file, args.output_file, model_name=args.model)

        elif args.command == 'evaluate':
            cli.evaluate_models(args.data_file)

        elif args.command == 'models':
            cli.show_model_info()

    except Exception as e:
        logger.error(f"Error executing command '{args.command}': {e}")
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()