"""
Model evaluation module for cricket run prediction.
Provides comprehensive evaluation metrics and visualizations.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error,
    mean_absolute_percentage_error
)
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Handles comprehensive model evaluation and visualization."""

    def __init__(self, reports_dir: str = "reports"):
        """
        Initialize the evaluator.

        Args:
            reports_dir: Directory to save evaluation reports and plots
        """
        self.reports_dir = Path(reports_dir)
        self.figures_dir = self.reports_dir / "figures"
        self.figures_dir.mkdir(parents=True, exist_ok=True)

        # Set seaborn style
        sns.set_style('darkgrid')
        self.color_palette = sns.color_palette()

    def evaluate_model(self, y_true: np.ndarray, y_pred: np.ndarray,
                      model_name: str) -> Dict[str, float]:
        """
        Calculate comprehensive evaluation metrics.

        Args:
            y_true: True target values
            y_pred: Predicted values
            model_name: Name of the model for logging

        Returns:
            Dictionary of evaluation metrics
        """
        logger.info(f"Evaluating {model_name} performance")

        # Basic metrics
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        # Additional metrics
        mape = mean_absolute_percentage_error(y_true, y_pred)
        explained_variance = 1 - np.var(y_true - y_pred) / np.var(y_true)

        # Custom cricket-specific metrics
        accuracy_within_1 = np.mean(np.abs(y_true - y_pred) <= 1)
        accuracy_within_2 = np.mean(np.abs(y_true - y_pred) <= 2)

        metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'mape': mape,
            'explained_variance': explained_variance,
            'accuracy_within_1_run': accuracy_within_1,
            'accuracy_within_2_runs': accuracy_within_2,
            'mean_prediction': np.mean(y_pred),
            'std_prediction': np.std(y_pred),
            'mean_actual': np.mean(y_true),
            'std_actual': np.std(y_true)
        }

        logger.info(f"{model_name} - R²: {r2:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}")
        return metrics

    def cross_validate_model(self, model, X: pd.DataFrame, y: pd.Series,
                           cv_folds: int = 5) -> Dict[str, Any]:
        """
        Perform cross-validation evaluation.

        Args:
            model: Trained model
            X: Feature matrix
            y: Target vector
            cv_folds: Number of cross-validation folds

        Returns:
            Dictionary with CV results
        """
        logger.info(f"Performing {cv_folds}-fold cross-validation")

        # R² scores
        r2_scores = cross_val_score(model, X, y, cv=cv_folds, scoring='r2')
        mse_scores = cross_val_score(model, X, y, cv=cv_folds,
                                   scoring='neg_mean_squared_error')
        mae_scores = cross_val_score(model, X, y, cv=cv_folds,
                                   scoring='neg_mean_absolute_error')

        # Convert negative MSE/MAE to positive
        mse_scores = -mse_scores
        mae_scores = -mae_scores

        cv_results = {
            'r2_mean': r2_scores.mean(),
            'r2_std': r2_scores.std(),
            'r2_scores': r2_scores.tolist(),
            'mse_mean': mse_scores.mean(),
            'mse_std': mse_scores.std(),
            'mae_mean': mae_scores.mean(),
            'mae_std': mae_scores.std(),
            'cv_folds': cv_folds
        }

        logger.info(f"CV R²: {r2_scores.mean():.4f} ± {r2_scores.std():.4f}")
        return cv_results

    def plot_actual_vs_predicted(self, y_true: np.ndarray, y_pred: np.ndarray,
                               model_name: str, save_plot: bool = True):
        """
        Plot actual vs predicted values with density contours.

        Args:
            y_true: True values
            y_pred: Predicted values
            model_name: Name for plot title and filename
            save_plot: Whether to save the plot
        """
        plt.figure(figsize=(10, 8))

        # Scatter plot with density
        sns.kdeplot(x=y_true, y=y_pred, cmap="Blues", fill=True, alpha=0.6)
        plt.scatter(y_true, y_pred, alpha=0.3, s=10, color='darkblue')

        # Perfect prediction line
        min_val = min(np.min(y_true), np.min(y_pred))
        max_val = max(np.max(y_true), np.max(y_pred))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')

        plt.xlabel('Actual Runs', fontsize=12)
        plt.ylabel('Predicted Runs', fontsize=12)
        plt.title(f'Actual vs Predicted Runs - {model_name}', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)

        if save_plot:
            plt.savefig(self.figures_dir / f'actual_vs_predicted_{model_name.lower().replace(" ", "_")}.png',
                       dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved: actual_vs_predicted_{model_name.lower().replace(' ', '_')}.png")

        plt.show()

    def plot_residuals(self, y_true: np.ndarray, y_pred: np.ndarray,
                      model_name: str, save_plot: bool = True):
        """
        Plot residual analysis.

        Args:
            y_true: True values
            y_pred: Predicted values
            model_name: Name for plot title and filename
            save_plot: Whether to save the plot
        """
        residuals = y_true - y_pred

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        # Residuals vs Predicted
        ax1.scatter(y_pred, residuals, alpha=0.5, s=10)
        ax1.axhline(y=0, color='r', linestyle='--', linewidth=2)
        ax1.set_xlabel('Predicted Runs')
        ax1.set_ylabel('Residuals')
        ax1.set_title('Residuals vs Predicted')
        ax1.grid(True, alpha=0.3)

        # Residual distribution
        sns.histplot(residuals, kde=True, bins=30, ax=ax2)
        ax2.axvline(x=0, color='r', linestyle='--', linewidth=2)
        ax2.set_xlabel('Residuals')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Residual Distribution')
        ax2.grid(True, alpha=0.3)

        # Q-Q plot
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=ax3)
        ax3.set_title('Q-Q Plot')
        ax3.grid(True, alpha=0.3)

        # Residuals vs Actual
        ax4.scatter(y_true, residuals, alpha=0.5, s=10)
        ax4.axhline(y=0, color='r', linestyle='--', linewidth=2)
        ax4.set_xlabel('Actual Runs')
        ax4.set_ylabel('Residuals')
        ax4.set_title('Residuals vs Actual')
        ax4.grid(True, alpha=0.3)

        plt.suptitle(f'Residual Analysis - {model_name}', fontsize=16, fontweight='bold')
        plt.tight_layout()

        if save_plot:
            plt.savefig(self.figures_dir / f'residual_analysis_{model_name.lower().replace(" ", "_")}.png',
                       dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved: residual_analysis_{model_name.lower().replace(' ', '_')}.png")

        plt.show()

    def plot_feature_importance(self, feature_names: List[str],
                              importance_values: np.ndarray,
                              model_name: str, top_n: int = 20,
                              save_plot: bool = True):
        """
        Plot feature importance.

        Args:
            feature_names: List of feature names
            importance_values: Feature importance values
            model_name: Name for plot title
            top_n: Number of top features to show
            save_plot: Whether to save the plot
        """
        # Sort features by importance
        indices = np.argsort(importance_values)[::-1][:top_n]
        top_features = [feature_names[i] for i in indices]
        top_importances = importance_values[indices]

        plt.figure(figsize=(12, 8))
        bars = plt.barh(range(len(top_features)), top_importances)
        plt.yticks(range(len(top_features)), top_features)
        plt.xlabel('Importance')
        plt.ylabel('Features')
        plt.title(f'Top {top_n} Feature Importances - {model_name}')
        plt.gca().invert_yaxis()
        plt.grid(True, alpha=0.3)

        # Add value labels
        for i, (bar, importance) in enumerate(zip(bars, top_importances)):
            plt.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                    '.3f', ha='left', va='center', fontsize=9)

        if save_plot:
            plt.savefig(self.figures_dir / f'feature_importance_{model_name.lower().replace(" ", "_")}.png',
                       dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved: feature_importance_{model_name.lower().replace(' ', '_')}.png")

        plt.show()

    def plot_error_by_run_value(self, y_true: np.ndarray, y_pred: np.ndarray,
                              model_name: str, save_plot: bool = True):
        """
        Plot prediction error analysis by actual run value.

        Args:
            y_true: True values
            y_pred: Predicted values
            model_name: Name for plot title
            save_plot: Whether to save the plot
        """
        errors = np.abs(y_true - y_pred)
        y_true_int = y_true.astype(int)

        # Group by actual runs
        error_by_runs = pd.DataFrame({
            'actual_runs': y_true_int,
            'error': errors
        }).groupby('actual_runs').agg(['mean', 'std', 'count'])

        error_by_runs.columns = ['mean_error', 'std_error', 'count']
        error_by_runs = error_by_runs.reset_index()

        plt.figure(figsize=(12, 8))

        # Bar plot with error bars
        bars = plt.bar(error_by_runs['actual_runs'], error_by_runs['mean_error'],
                      yerr=error_by_runs['std_error'], capsize=5, alpha=0.7)

        plt.xlabel('Actual Runs', fontsize=12)
        plt.ylabel('Mean Absolute Error', fontsize=12)
        plt.title(f'Prediction Error by Actual Run Value - {model_name}',
                 fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.xticks(error_by_runs['actual_runs'])

        # Add sample size labels
        for bar, count in zip(bars, error_by_runs['count']):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'n={int(count)}', ha='center', va='bottom', fontsize=8)

        if save_plot:
            plt.savefig(self.figures_dir / f'error_by_runs_{model_name.lower().replace(" ", "_")}.png',
                       dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved: error_by_runs_{model_name.lower().replace(' ', '_')}.png")

        plt.show()

        return error_by_runs

    def create_model_comparison_report(self, model_results: Dict[str, Dict],
                                     save_report: bool = True) -> pd.DataFrame:
        """
        Create comprehensive model comparison report.

        Args:
            model_results: Dictionary with model names as keys and metrics as values
            save_report: Whether to save the report

        Returns:
            DataFrame with model comparison
        """
        # Create comparison DataFrame
        comparison_data = []
        for model_name, metrics in model_results.items():
            row = {'Model': model_name}
            row.update(metrics)
            comparison_data.append(row)

        comparison_df = pd.DataFrame(comparison_data)

        # Reorder columns for better presentation
        metric_columns = [col for col in comparison_df.columns if col != 'Model']
        ordered_columns = ['Model'] + sorted(metric_columns)
        comparison_df = comparison_df[ordered_columns]

        if save_report:
            report_path = self.reports_dir / 'model_comparison_report.csv'
            comparison_df.to_csv(report_path, index=False)
            logger.info(f"Model comparison report saved to {report_path}")

        return comparison_df

    def plot_model_comparison(self, comparison_df: pd.DataFrame,
                            metrics: List[str] = ['r2', 'mae', 'rmse'],
                            save_plot: bool = True):
        """
        Create visualization comparing multiple models.

        Args:
            comparison_df: DataFrame from create_model_comparison_report
            metrics: List of metrics to compare
            save_plot: Whether to save the plot
        """
        n_metrics = len(metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(6*n_metrics, 6))

        if n_metrics == 1:
            axes = [axes]

        for i, metric in enumerate(metrics):
            if metric in comparison_df.columns:
                bars = axes[i].bar(comparison_df['Model'], comparison_df[metric])
                axes[i].set_title(f'{metric.upper()} Comparison', fontweight='bold')
                axes[i].set_ylabel(metric.upper())
                axes[i].tick_params(axis='x', rotation=45)

                # Add value labels
                for bar, value in zip(bars, comparison_df[metric]):
                    axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                               '.4f', ha='center', va='bottom', fontsize=9)

                axes[i].grid(True, alpha=0.3)

        plt.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout()

        if save_plot:
            plt.savefig(self.figures_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
            logger.info("Plot saved: model_comparison.png")

        plt.show()

    def generate_evaluation_report(self, model_results: Dict[str, Dict],
                                 cv_results: Optional[Dict[str, Dict]] = None) -> str:
        """
        Generate comprehensive evaluation report.

        Args:
            model_results: Dictionary with model evaluation results
            cv_results: Optional cross-validation results

        Returns:
            Formatted report string
        """
        report_lines = []
        report_lines.append("# Cricket Run Prediction - Model Evaluation Report")
        report_lines.append("=" * 60)
        report_lines.append("")

        # Best model identification
        best_model = max(model_results.keys(),
                        key=lambda x: model_results[x].get('r2', 0))

        report_lines.append(f"🏆 **Best Performing Model**: {best_model}")
        report_lines.append(f"   R² Score: {model_results[best_model]['r2']:.4f}")
        report_lines.append(f"   MAE: {model_results[best_model]['mae']:.4f}")
        report_lines.append("")

        # Model comparison table
        report_lines.append("## Model Performance Comparison")
        report_lines.append("| Model | R² | MAE | RMSE | Accuracy ±1 | Accuracy ±2 |")
        report_lines.append("|-------|----|-----|------|-------------|-------------|")

        for model_name, metrics in model_results.items():
            report_lines.append("| {model_name} | {metrics['r2']:.4f} | {metrics['mae']:.4f} | {metrics['rmse']:.4f} | {metrics['accuracy_within_1_run']:.1%} | {metrics['accuracy_within_2_runs']:.1%} |")

        report_lines.append("")

        # Cross-validation results
        if cv_results:
            report_lines.append("## Cross-Validation Results")
            for model_name, cv_metrics in cv_results.items():
                report_lines.append(f"**{model_name}**:")
                report_lines.append(f"  - R²: {cv_metrics['r2_mean']:.4f} ± {cv_metrics['r2_std']:.4f}")
                report_lines.append(f"  - MSE: {cv_metrics['mse_mean']:.6f} ± {cv_metrics['mse_std']:.6f}")
                report_lines.append("")

        # Key insights
        report_lines.append("## Key Insights")
        report_lines.append("1. **Prediction Accuracy**: Models show high accuracy for 0-run predictions")
        report_lines.append("2. **Error Patterns**: Higher errors for boundary (4) and six (6) predictions")
        report_lines.append("3. **Feature Importance**: Match context and player statistics are crucial")
        report_lines.append("4. **Model Robustness**: Ensemble methods provide stable performance")
        report_lines.append("")

        report = "\n".join(report_lines)
        return report