# Ball-by-Ball Cricket Run Prediction
# Production-grade machine learning pipeline for predicting cricket runs

__version__ = "1.0.0"
__author__ = "Beyond Instinct Team"
__description__ = "Advanced ML system for ball-by-ball cricket run prediction"

# Lazy imports to avoid heavy dependencies at package level
def __getattr__(name):
    if name == 'CricketDataLoader':
        from .data_loader import CricketDataLoader
        return CricketDataLoader
    elif name == 'CricketDataPreprocessor':
        from .preprocessor import CricketDataPreprocessor
        return CricketDataPreprocessor
    elif name == 'CricketFeatureEngineer':
        from .feature_engineering import CricketFeatureEngineer
        return CricketFeatureEngineer
    elif name == 'CricketModelTrainer':
        from .model_trainer import CricketModelTrainer
        return CricketModelTrainer
    elif name == 'ModelEvaluator':
        from .evaluator import ModelEvaluator
        return ModelEvaluator
    elif name == 'RunPredictor':
        from .predictor import RunPredictor
        return RunPredictor
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = [
    'CricketDataLoader',
    'CricketDataPreprocessor',
    'CricketFeatureEngineer',
    'CricketModelTrainer',
    'ModelEvaluator',
    'RunPredictor'
]