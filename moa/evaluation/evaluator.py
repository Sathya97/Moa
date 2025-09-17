"""
Main evaluation orchestrator for MoA prediction models.

This module provides comprehensive model evaluation including
cross-validation, statistical testing, and comparison analysis.
"""

import os
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import make_scorer
import joblib

from moa.models.multimodal_model import MultiModalMoAPredictor
from moa.evaluation.metrics import MoAMetrics
from moa.evaluation.statistical_tests import StatisticalTests
from moa.evaluation.visualization import EvaluationVisualizer
from moa.training.data_loader import MoADataLoader
from moa.utils.config import Config
from moa.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class EvaluationResult:
    """Container for evaluation results."""
    model_name: str
    metrics: Dict[str, float]
    predictions: np.ndarray
    probabilities: Optional[np.ndarray]
    true_labels: np.ndarray
    evaluation_time: float
    metadata: Dict[str, Any]


class MoAEvaluator:
    """Comprehensive evaluator for MoA prediction models."""
    
    def __init__(
        self,
        config: Config,
        moa_classes: List[str],
        output_dir: Optional[Union[str, Path]] = None
    ):
        """
        Initialize MoA evaluator.
        
        Args:
            config: Configuration object
            moa_classes: List of MoA class names
            output_dir: Directory to save evaluation results
        """
        self.config = config
        self.moa_classes = moa_classes
        self.output_dir = Path(output_dir) if output_dir else Path("evaluation_results")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.metrics_calculator = MoAMetrics(moa_classes)
        self.statistical_tests = StatisticalTests()
        self.visualizer = EvaluationVisualizer(self.output_dir)
        
        # Evaluation configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.save_predictions = config.get("evaluation.save_predictions", True)
        self.generate_plots = config.get("evaluation.generate_plots", True)
        
        logger.info(f"Initialized MoA evaluator for {len(moa_classes)} classes")
        logger.info(f"Output directory: {self.output_dir}")
    
    def evaluate_model(
        self,
        model: Union[MultiModalMoAPredictor, nn.Module],
        data_loader,
        model_name: str = "model",
        return_predictions: bool = True
    ) -> EvaluationResult:
        """
        Evaluate a single model on given data.
        
        Args:
            model: Model to evaluate
            data_loader: Data loader for evaluation
            model_name: Name identifier for the model
            return_predictions: Whether to return predictions
            
        Returns:
            Evaluation result object
        """
        logger.info(f"Evaluating model: {model_name}")
        start_time = time.time()
        
        # Set model to evaluation mode
        model.eval()
        model.to(self.device)
        
        # Collect predictions and targets
        all_predictions = []
        all_probabilities = []
        all_targets = []
        
        with torch.no_grad():
            for batch_idx, (batch_data, targets) in enumerate(data_loader):
                # Move data to device
                targets = targets.to(self.device)
                if isinstance(batch_data, dict):
                    for key, value in batch_data.items():
                        if hasattr(value, 'to'):
                            batch_data[key] = value.to(self.device)
                
                # Get model predictions
                if hasattr(model, 'predict'):
                    # Multi-modal model
                    probabilities = model.predict(batch_data, return_probabilities=True)
                    predictions = (probabilities > 0.5).float()
                else:
                    # Standard PyTorch model
                    logits = model(batch_data)
                    probabilities = torch.sigmoid(logits)
                    predictions = (probabilities > 0.5).float()
                
                # Collect results
                all_predictions.append(predictions.cpu())
                all_probabilities.append(probabilities.cpu())
                all_targets.append(targets.cpu())
                
                if batch_idx % 100 == 0:
                    logger.debug(f"Processed {batch_idx + 1} batches")
        
        # Concatenate all results
        predictions = torch.cat(all_predictions, dim=0).numpy()
        probabilities = torch.cat(all_probabilities, dim=0).numpy()
        targets = torch.cat(all_targets, dim=0).numpy()
        
        # Compute metrics
        metrics = self.metrics_calculator.compute_moa_specific_metrics(
            targets, predictions, probabilities
        )
        
        evaluation_time = time.time() - start_time
        
        # Create evaluation result
        result = EvaluationResult(
            model_name=model_name,
            metrics=metrics,
            predictions=predictions if return_predictions else None,
            probabilities=probabilities if return_predictions else None,
            true_labels=targets if return_predictions else None,
            evaluation_time=evaluation_time,
            metadata={
                'num_samples': len(targets),
                'num_classes': len(self.moa_classes),
                'device': str(self.device)
            }
        )
        
        # Save results
        self._save_evaluation_result(result)
        
        # Generate visualizations
        if self.generate_plots and return_predictions:
            self._generate_evaluation_plots(result)
        
        logger.info(f"Evaluation completed in {evaluation_time:.2f}s")
        logger.info(f"Key metrics - AUROC: {metrics.get('auroc_macro', 0):.4f}, "
                   f"F1: {metrics.get('f1_macro', 0):.4f}")
        
        return result
    
    def compare_models(
        self,
        models: Dict[str, Union[MultiModalMoAPredictor, nn.Module]],
        data_loader,
        statistical_test: str = "wilcoxon"
    ) -> Dict[str, Any]:
        """
        Compare multiple models on the same dataset.
        
        Args:
            models: Dictionary of model_name -> model
            data_loader: Data loader for evaluation
            statistical_test: Statistical test for comparison
            
        Returns:
            Comparison results
        """
        logger.info(f"Comparing {len(models)} models")
        
        # Evaluate all models
        results = {}
        for model_name, model in models.items():
            results[model_name] = self.evaluate_model(
                model, data_loader, model_name, return_predictions=True
            )
        
        # Create comparison summary
        comparison_summary = self._create_comparison_summary(results)
        
        # Statistical significance testing
        if len(models) > 1:
            significance_results = self._perform_statistical_tests(
                results, statistical_test
            )
            comparison_summary['statistical_tests'] = significance_results
        
        # Generate comparison visualizations
        if self.generate_plots:
            self._generate_comparison_plots(results)
        
        # Save comparison results
        self._save_comparison_results(comparison_summary)
        
        return comparison_summary
    
    def cross_validate_model(
        self,
        model_class,
        model_kwargs: Dict[str, Any],
        dataset,
        cv_folds: int = 5,
        stratify: bool = True
    ) -> Dict[str, Any]:
        """
        Perform cross-validation evaluation.
        
        Args:
            model_class: Model class to instantiate
            model_kwargs: Keyword arguments for model initialization
            dataset: Dataset for cross-validation
            cv_folds: Number of CV folds
            stratify: Whether to use stratified CV
            
        Returns:
            Cross-validation results
        """
        logger.info(f"Performing {cv_folds}-fold cross-validation")
        
        # Prepare data for sklearn-style CV
        X, y = self._prepare_cv_data(dataset)
        
        # Create CV splitter
        if stratify:
            # For multi-label, use iterative stratification
            cv_splitter = self._create_multilabel_cv_splitter(y, cv_folds)
        else:
            cv_splitter = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        # Perform cross-validation
        cv_results = []
        for fold, (train_idx, val_idx) in enumerate(cv_splitter.split(X, y)):
            logger.info(f"Training fold {fold + 1}/{cv_folds}")
            
            # Create fold-specific data loaders
            train_subset = torch.utils.data.Subset(dataset, train_idx)
            val_subset = torch.utils.data.Subset(dataset, val_idx)
            
            train_loader = torch.utils.data.DataLoader(train_subset, batch_size=32, shuffle=True)
            val_loader = torch.utils.data.DataLoader(val_subset, batch_size=64, shuffle=False)
            
            # Initialize and train model
            model = model_class(**model_kwargs)
            
            # Train model (simplified - in practice would use full training pipeline)
            # This is a placeholder for the actual training loop
            fold_result = self._train_and_evaluate_fold(model, train_loader, val_loader, fold)
            cv_results.append(fold_result)
        
        # Aggregate CV results
        cv_summary = self._aggregate_cv_results(cv_results)
        
        # Save CV results
        self._save_cv_results(cv_summary)
        
        return cv_summary
    
    def evaluate_on_external_dataset(
        self,
        model: Union[MultiModalMoAPredictor, nn.Module],
        external_data_path: Union[str, Path],
        dataset_name: str = "external"
    ) -> EvaluationResult:
        """
        Evaluate model on external dataset.
        
        Args:
            model: Trained model
            external_data_path: Path to external dataset
            dataset_name: Name of external dataset
            
        Returns:
            Evaluation result
        """
        logger.info(f"Evaluating on external dataset: {dataset_name}")
        
        # Load external dataset
        external_loader = self._load_external_dataset(external_data_path)
        
        # Evaluate model
        result = self.evaluate_model(
            model, external_loader, f"model_on_{dataset_name}", return_predictions=True
        )
        
        # Add external dataset metadata
        result.metadata['dataset_name'] = dataset_name
        result.metadata['dataset_path'] = str(external_data_path)
        
        return result
    
    def _create_comparison_summary(self, results: Dict[str, EvaluationResult]) -> Dict[str, Any]:
        """Create summary of model comparison."""
        summary = {
            'models': list(results.keys()),
            'metrics_comparison': {},
            'best_model': {},
            'evaluation_times': {}
        }
        
        # Extract metrics for comparison
        all_metrics = {}
        for model_name, result in results.items():
            all_metrics[model_name] = result.metrics
            summary['evaluation_times'][model_name] = result.evaluation_time
        
        # Create metrics comparison table
        metrics_df = pd.DataFrame(all_metrics).T
        summary['metrics_comparison'] = metrics_df.to_dict()
        
        # Find best model for each metric
        for metric in metrics_df.columns:
            if 'loss' in metric.lower():
                best_model = metrics_df[metric].idxmin()
            else:
                best_model = metrics_df[metric].idxmax()
            summary['best_model'][metric] = best_model
        
        return summary
    
    def _perform_statistical_tests(
        self,
        results: Dict[str, EvaluationResult],
        test_type: str = "wilcoxon"
    ) -> Dict[str, Any]:
        """Perform statistical significance tests."""
        significance_results = {}
        
        # Get predictions for all models
        model_predictions = {}
        true_labels = None
        
        for model_name, result in results.items():
            model_predictions[model_name] = result.probabilities
            if true_labels is None:
                true_labels = result.true_labels
        
        # Pairwise comparisons
        model_names = list(model_predictions.keys())
        for i in range(len(model_names)):
            for j in range(i + 1, len(model_names)):
                model1, model2 = model_names[i], model_names[j]
                
                # Perform statistical test
                test_result = self.statistical_tests.compare_models(
                    model_predictions[model1],
                    model_predictions[model2],
                    true_labels,
                    test_type=test_type
                )
                
                comparison_key = f"{model1}_vs_{model2}"
                significance_results[comparison_key] = test_result
        
        return significance_results
    
    def _save_evaluation_result(self, result: EvaluationResult):
        """Save evaluation result to file."""
        if not self.save_predictions:
            return
        
        # Save metrics
        metrics_file = self.output_dir / f"{result.model_name}_metrics.json"
        import json
        with open(metrics_file, 'w') as f:
            json.dump(result.metrics, f, indent=2)
        
        # Save predictions and targets
        if result.predictions is not None:
            predictions_file = self.output_dir / f"{result.model_name}_predictions.npz"
            np.savez(
                predictions_file,
                predictions=result.predictions,
                probabilities=result.probabilities,
                true_labels=result.true_labels
            )
        
        logger.debug(f"Saved evaluation results for {result.model_name}")
    
    def _generate_evaluation_plots(self, result: EvaluationResult):
        """Generate evaluation plots."""
        try:
            self.visualizer.plot_roc_curves(
                result.true_labels,
                result.probabilities,
                self.moa_classes,
                title=f"ROC Curves - {result.model_name}"
            )
            
            self.visualizer.plot_precision_recall_curves(
                result.true_labels,
                result.probabilities,
                self.moa_classes,
                title=f"PR Curves - {result.model_name}"
            )
            
            self.visualizer.plot_confusion_matrices(
                result.true_labels,
                result.predictions,
                self.moa_classes,
                title=f"Confusion Matrices - {result.model_name}"
            )
            
        except Exception as e:
            logger.warning(f"Error generating evaluation plots: {e}")
    
    def _generate_comparison_plots(self, results: Dict[str, EvaluationResult]):
        """Generate model comparison plots."""
        try:
            self.visualizer.plot_model_comparison(
                results,
                metrics=['auroc_macro', 'f1_macro', 'precision_macro', 'recall_macro']
            )
        except Exception as e:
            logger.warning(f"Error generating comparison plots: {e}")
    
    def _save_comparison_results(self, comparison_summary: Dict[str, Any]):
        """Save model comparison results."""
        comparison_file = self.output_dir / "model_comparison.json"
        import json
        with open(comparison_file, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            serializable_summary = self._make_json_serializable(comparison_summary)
            json.dump(serializable_summary, f, indent=2)
    
    def _make_json_serializable(self, obj):
        """Make object JSON serializable."""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        else:
            return obj

    def _prepare_cv_data(self, dataset):
        """Prepare data for cross-validation."""
        # Extract features and labels from dataset
        # This is a simplified version - in practice would handle multi-modal data
        X = []
        y = []

        for i in range(len(dataset)):
            batch_data, target = dataset[i]
            # Flatten all features into a single vector (simplified)
            features = []
            for key, value in batch_data.items():
                if hasattr(value, 'flatten'):
                    features.extend(value.flatten().tolist())
            X.append(features)
            y.append(target.numpy())

        return np.array(X), np.array(y)

    def _create_multilabel_cv_splitter(self, y, n_splits):
        """Create stratified splitter for multi-label data."""
        from sklearn.model_selection import KFold
        # Simplified - use regular KFold for now
        # In practice, would use iterative stratification for multi-label
        return KFold(n_splits=n_splits, shuffle=True, random_state=42)

    def _train_and_evaluate_fold(self, model, train_loader, val_loader, fold):
        """Train and evaluate model for one CV fold."""
        # Placeholder for actual training
        # In practice, would use the full training pipeline

        # Simplified evaluation
        result = self.evaluate_model(
            model, val_loader, f"fold_{fold}", return_predictions=True
        )

        return {
            'fold': fold,
            'metrics': result.metrics,
            'evaluation_time': result.evaluation_time
        }

    def _aggregate_cv_results(self, cv_results):
        """Aggregate cross-validation results."""
        # Collect metrics from all folds
        all_metrics = {}
        for result in cv_results:
            for metric, value in result['metrics'].items():
                if metric not in all_metrics:
                    all_metrics[metric] = []
                all_metrics[metric].append(value)

        # Compute statistics
        cv_summary = {
            'num_folds': len(cv_results),
            'metrics_mean': {},
            'metrics_std': {},
            'metrics_min': {},
            'metrics_max': {},
            'fold_results': cv_results
        }

        for metric, values in all_metrics.items():
            cv_summary['metrics_mean'][metric] = np.mean(values)
            cv_summary['metrics_std'][metric] = np.std(values)
            cv_summary['metrics_min'][metric] = np.min(values)
            cv_summary['metrics_max'][metric] = np.max(values)

        return cv_summary

    def _save_cv_results(self, cv_summary):
        """Save cross-validation results."""
        cv_file = self.output_dir / "cross_validation_results.json"
        import json
        with open(cv_file, 'w') as f:
            serializable_summary = self._make_json_serializable(cv_summary)
            json.dump(serializable_summary, f, indent=2)

    def _load_external_dataset(self, data_path):
        """Load external dataset."""
        # Placeholder - would implement actual external data loading
        # For now, return None to indicate not implemented
        logger.warning("External dataset loading not implemented")
        return None
