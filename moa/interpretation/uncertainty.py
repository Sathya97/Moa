"""
Uncertainty estimation and calibration for MoA prediction models.

This module implements various uncertainty quantification methods including
Monte Carlo dropout, ensemble predictions, and calibration techniques.
"""

from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from moa.utils.config import Config
from moa.utils.logger import get_logger

logger = get_logger(__name__)


class UncertaintyEstimator:
    """
    Uncertainty estimator for MoA prediction models.
    
    Implements multiple uncertainty quantification methods:
    - Monte Carlo Dropout
    - Deep Ensembles
    - Bayesian Neural Networks (approximate)
    - Calibration techniques
    - Epistemic vs Aleatoric uncertainty decomposition
    """
    
    def __init__(self, model: nn.Module, config: Config):
        """
        Initialize uncertainty estimator.
        
        Args:
            model: MoA prediction model
            config: Configuration object
        """
        self.model = model
        self.config = config
        
        # Uncertainty estimation parameters
        self.mc_samples = config.get('interpretation.uncertainty.mc_samples', 100)
        self.dropout_rate = config.get('interpretation.uncertainty.dropout_rate', 0.1)
        self.ensemble_size = config.get('interpretation.uncertainty.ensemble_size', 5)
        self.calibration_method = config.get('interpretation.uncertainty.calibration_method', 'platt')
        
        # Calibration models
        self.calibration_models = {}
        self.is_calibrated = False
        
        logger.info("Uncertainty estimator initialized")
    
    def estimate_uncertainty(
        self,
        batch_data: Dict[str, torch.Tensor],
        methods: List[str] = ['mc_dropout', 'ensemble'],
        return_samples: bool = False
    ) -> Dict[str, Any]:
        """
        Estimate prediction uncertainty using multiple methods.
        
        Args:
            batch_data: Input batch data
            methods: List of uncertainty estimation methods
            return_samples: Whether to return individual samples
            
        Returns:
            Dictionary containing uncertainty estimates
        """
        logger.info(f"Estimating uncertainty using methods: {methods}")
        
        uncertainty_results = {
            'methods': {},
            'summary': {},
            'calibration': {}
        }
        
        # Compute uncertainty for each method
        for method in methods:
            if method == 'mc_dropout':
                uncertainty_results['methods']['mc_dropout'] = self._monte_carlo_dropout(
                    batch_data, return_samples
                )
            elif method == 'ensemble':
                uncertainty_results['methods']['ensemble'] = self._ensemble_uncertainty(
                    batch_data, return_samples
                )
            elif method == 'bayesian':
                uncertainty_results['methods']['bayesian'] = self._bayesian_uncertainty(
                    batch_data, return_samples
                )
            else:
                logger.warning(f"Unknown uncertainty method: {method}")
        
        # Create uncertainty summary
        uncertainty_results['summary'] = self._create_uncertainty_summary(
            uncertainty_results['methods']
        )
        
        return uncertainty_results
    
    def _monte_carlo_dropout(
        self,
        batch_data: Dict[str, torch.Tensor],
        return_samples: bool = False
    ) -> Dict[str, Any]:
        """Estimate uncertainty using Monte Carlo Dropout."""
        self.model.train()  # Enable dropout
        
        predictions = []
        logits_samples = []
        
        with torch.no_grad():
            for _ in range(self.mc_samples):
                # Forward pass with dropout
                output = self.model(batch_data)
                logits = output['logits']
                pred = torch.sigmoid(logits)
                
                predictions.append(pred.cpu().numpy())
                if return_samples:
                    logits_samples.append(logits.cpu().numpy())
        
        self.model.eval()  # Disable dropout
        
        # Convert to numpy arrays
        predictions = np.array(predictions)  # [mc_samples, batch_size, num_classes]
        
        # Compute uncertainty statistics
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)
        
        # Epistemic uncertainty (model uncertainty)
        epistemic_uncertainty = std_pred
        
        # Predictive entropy
        mean_pred_clipped = np.clip(mean_pred, 1e-8, 1 - 1e-8)
        predictive_entropy = -np.sum(
            mean_pred_clipped * np.log(mean_pred_clipped) + 
            (1 - mean_pred_clipped) * np.log(1 - mean_pred_clipped),
            axis=-1
        )
        
        # Mutual information (epistemic uncertainty measure)
        entropies = []
        for i in range(self.mc_samples):
            pred_clipped = np.clip(predictions[i], 1e-8, 1 - 1e-8)
            entropy = -np.sum(
                pred_clipped * np.log(pred_clipped) + 
                (1 - pred_clipped) * np.log(1 - pred_clipped),
                axis=-1
            )
            entropies.append(entropy)
        
        mean_entropy = np.mean(entropies, axis=0)
        mutual_information = predictive_entropy - mean_entropy
        
        result = {
            'method': 'mc_dropout',
            'mean_prediction': mean_pred,
            'std_prediction': std_pred,
            'epistemic_uncertainty': epistemic_uncertainty,
            'predictive_entropy': predictive_entropy,
            'mutual_information': mutual_information,
            'mc_samples': self.mc_samples
        }
        
        if return_samples:
            result['prediction_samples'] = predictions
            result['logits_samples'] = np.array(logits_samples) if logits_samples else None
        
        return result
    
    def _ensemble_uncertainty(
        self,
        batch_data: Dict[str, torch.Tensor],
        return_samples: bool = False
    ) -> Dict[str, Any]:
        """Estimate uncertainty using ensemble predictions."""
        # Note: This is a simplified implementation
        # In practice, you would have multiple trained models
        
        logger.warning("Ensemble uncertainty is simplified - requires multiple trained models")
        
        # Simulate ensemble by adding noise to model parameters
        predictions = []
        original_state = {}
        
        # Save original parameters
        for name, param in self.model.named_parameters():
            original_state[name] = param.data.clone()
        
        self.model.eval()
        
        with torch.no_grad():
            for i in range(self.ensemble_size):
                # Add small noise to parameters (simulating different models)
                noise_scale = 0.01
                for name, param in self.model.named_parameters():
                    if param.requires_grad:
                        noise = torch.randn_like(param) * noise_scale
                        param.data.add_(noise)
                
                # Forward pass
                output = self.model(batch_data)
                pred = torch.sigmoid(output['logits'])
                predictions.append(pred.cpu().numpy())
                
                # Restore original parameters
                for name, param in self.model.named_parameters():
                    param.data.copy_(original_state[name])
        
        # Convert to numpy arrays
        predictions = np.array(predictions)  # [ensemble_size, batch_size, num_classes]
        
        # Compute uncertainty statistics
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)
        
        # Ensemble disagreement
        disagreement = std_pred
        
        # Predictive entropy
        mean_pred_clipped = np.clip(mean_pred, 1e-8, 1 - 1e-8)
        predictive_entropy = -np.sum(
            mean_pred_clipped * np.log(mean_pred_clipped) + 
            (1 - mean_pred_clipped) * np.log(1 - mean_pred_clipped),
            axis=-1
        )
        
        result = {
            'method': 'ensemble',
            'mean_prediction': mean_pred,
            'std_prediction': std_pred,
            'ensemble_disagreement': disagreement,
            'predictive_entropy': predictive_entropy,
            'ensemble_size': self.ensemble_size
        }
        
        if return_samples:
            result['prediction_samples'] = predictions
        
        return result
    
    def _bayesian_uncertainty(
        self,
        batch_data: Dict[str, torch.Tensor],
        return_samples: bool = False
    ) -> Dict[str, Any]:
        """Estimate uncertainty using Bayesian approximation."""
        # Simplified Bayesian uncertainty using variational inference approximation
        # In practice, this would require a proper Bayesian neural network
        
        logger.warning("Bayesian uncertainty is simplified - requires proper BNN implementation")
        
        # Use MC Dropout as approximation to Bayesian inference
        mc_result = self._monte_carlo_dropout(batch_data, return_samples)
        
        # Add Bayesian-specific metrics
        result = mc_result.copy()
        result['method'] = 'bayesian_approximate'
        
        # Approximate aleatoric uncertainty (data uncertainty)
        # This would require the model to predict both mean and variance
        aleatoric_uncertainty = np.ones_like(mc_result['mean_prediction']) * 0.1  # Placeholder
        
        result['aleatoric_uncertainty'] = aleatoric_uncertainty
        result['total_uncertainty'] = np.sqrt(
            mc_result['epistemic_uncertainty']**2 + aleatoric_uncertainty**2
        )
        
        return result
    
    def calibrate_predictions(
        self,
        predictions: np.ndarray,
        true_labels: np.ndarray,
        method: str = 'platt'
    ) -> Dict[str, Any]:
        """
        Calibrate model predictions to improve uncertainty estimates.
        
        Args:
            predictions: Model predictions [batch_size, num_classes]
            true_labels: True binary labels [batch_size, num_classes]
            method: Calibration method ('platt', 'isotonic', 'temperature')
            
        Returns:
            Dictionary containing calibration results
        """
        logger.info(f"Calibrating predictions using {method} method")
        
        calibration_results = {
            'method': method,
            'calibration_models': {},
            'calibration_metrics': {},
            'calibrated_predictions': np.zeros_like(predictions)
        }
        
        num_classes = predictions.shape[1]
        
        # Calibrate each class separately
        for class_idx in range(num_classes):
            class_preds = predictions[:, class_idx]
            class_labels = true_labels[:, class_idx]
            
            if method == 'platt':
                # Platt scaling (logistic regression)
                calibrator = LogisticRegression()
                calibrator.fit(class_preds.reshape(-1, 1), class_labels)
                calibrated_preds = calibrator.predict_proba(class_preds.reshape(-1, 1))[:, 1]
                
            elif method == 'isotonic':
                # Isotonic regression
                calibrator = IsotonicRegression(out_of_bounds='clip')
                calibrator.fit(class_preds, class_labels)
                calibrated_preds = calibrator.predict(class_preds)
                
            elif method == 'temperature':
                # Temperature scaling
                calibrator = self._fit_temperature_scaling(class_preds, class_labels)
                calibrated_preds = self._apply_temperature_scaling(class_preds, calibrator)
                
            else:
                raise ValueError(f"Unknown calibration method: {method}")
            
            calibration_results['calibration_models'][class_idx] = calibrator
            calibration_results['calibrated_predictions'][:, class_idx] = calibrated_preds
            
            # Compute calibration metrics for this class
            calibration_results['calibration_metrics'][class_idx] = self._compute_calibration_metrics(
                class_preds, calibrated_preds, class_labels
            )
        
        # Store calibration models for future use
        self.calibration_models[method] = calibration_results['calibration_models']
        self.is_calibrated = True
        
        return calibration_results
    
    def _fit_temperature_scaling(self, predictions: np.ndarray, labels: np.ndarray) -> float:
        """Fit temperature scaling parameter."""
        # Convert to logits (inverse sigmoid)
        predictions_clipped = np.clip(predictions, 1e-8, 1 - 1e-8)
        logits = np.log(predictions_clipped / (1 - predictions_clipped))
        
        # Find optimal temperature using cross-entropy loss
        def temperature_loss(temperature):
            scaled_logits = logits / temperature
            scaled_probs = 1 / (1 + np.exp(-scaled_logits))
            loss = -np.mean(labels * np.log(scaled_probs + 1e-8) + 
                           (1 - labels) * np.log(1 - scaled_probs + 1e-8))
            return loss
        
        # Simple grid search for temperature
        temperatures = np.linspace(0.1, 5.0, 50)
        losses = [temperature_loss(t) for t in temperatures]
        optimal_temperature = temperatures[np.argmin(losses)]
        
        return optimal_temperature
    
    def _apply_temperature_scaling(self, predictions: np.ndarray, temperature: float) -> np.ndarray:
        """Apply temperature scaling to predictions."""
        predictions_clipped = np.clip(predictions, 1e-8, 1 - 1e-8)
        logits = np.log(predictions_clipped / (1 - predictions_clipped))
        scaled_logits = logits / temperature
        scaled_probs = 1 / (1 + np.exp(-scaled_logits))
        return scaled_probs
    
    def _compute_calibration_metrics(
        self,
        original_preds: np.ndarray,
        calibrated_preds: np.ndarray,
        true_labels: np.ndarray
    ) -> Dict[str, float]:
        """Compute calibration metrics."""
        metrics = {}
        
        # Expected Calibration Error (ECE)
        metrics['ece_original'] = self._expected_calibration_error(original_preds, true_labels)
        metrics['ece_calibrated'] = self._expected_calibration_error(calibrated_preds, true_labels)
        
        # Maximum Calibration Error (MCE)
        metrics['mce_original'] = self._maximum_calibration_error(original_preds, true_labels)
        metrics['mce_calibrated'] = self._maximum_calibration_error(calibrated_preds, true_labels)
        
        # Brier Score
        metrics['brier_original'] = np.mean((original_preds - true_labels)**2)
        metrics['brier_calibrated'] = np.mean((calibrated_preds - true_labels)**2)
        
        return metrics
    
    def _expected_calibration_error(self, predictions: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
        """Compute Expected Calibration Error."""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (predictions > bin_lower) & (predictions <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = labels[in_bin].mean()
                avg_confidence_in_bin = predictions[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece
    
    def _maximum_calibration_error(self, predictions: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
        """Compute Maximum Calibration Error."""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        mce = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (predictions > bin_lower) & (predictions <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = labels[in_bin].mean()
                avg_confidence_in_bin = predictions[in_bin].mean()
                mce = max(mce, np.abs(avg_confidence_in_bin - accuracy_in_bin))
        
        return mce
    
    def _create_uncertainty_summary(self, methods_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create summary of uncertainty estimation results."""
        summary = {
            'uncertainty_metrics': {},
            'confidence_intervals': {},
            'uncertainty_ranking': []
        }
        
        # Aggregate uncertainty metrics across methods
        for method_name, method_results in methods_results.items():
            if 'mean_prediction' in method_results:
                mean_pred = method_results['mean_prediction']
                std_pred = method_results['std_prediction']
                
                # Compute confidence intervals
                confidence_lower = mean_pred - 1.96 * std_pred
                confidence_upper = mean_pred + 1.96 * std_pred
                
                summary['confidence_intervals'][method_name] = {
                    'lower': confidence_lower,
                    'upper': confidence_upper,
                    'width': confidence_upper - confidence_lower
                }
                
                # Compute uncertainty metrics
                summary['uncertainty_metrics'][method_name] = {
                    'mean_uncertainty': np.mean(std_pred),
                    'max_uncertainty': np.max(std_pred),
                    'uncertainty_std': np.std(std_pred)
                }
                
                if 'predictive_entropy' in method_results:
                    summary['uncertainty_metrics'][method_name]['mean_entropy'] = np.mean(
                        method_results['predictive_entropy']
                    )
        
        return summary
    
    def visualize_uncertainty(
        self,
        uncertainty_results: Dict[str, Any],
        output_path: Optional[str] = None,
        sample_idx: int = 0
    ) -> plt.Figure:
        """
        Visualize uncertainty estimation results.
        
        Args:
            uncertainty_results: Results from estimate_uncertainty
            output_path: Path to save the visualization
            sample_idx: Sample index to visualize
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Prediction uncertainty by method
        ax = axes[0, 0]
        methods = list(uncertainty_results['methods'].keys())
        uncertainties = []
        
        for method in methods:
            if 'std_prediction' in uncertainty_results['methods'][method]:
                std_pred = uncertainty_results['methods'][method]['std_prediction']
                uncertainties.append(np.mean(std_pred[sample_idx]))
            else:
                uncertainties.append(0)
        
        ax.bar(methods, uncertainties)
        ax.set_title('Prediction Uncertainty by Method')
        ax.set_ylabel('Mean Uncertainty')
        ax.tick_params(axis='x', rotation=45)
        
        # Plot 2: Uncertainty distribution
        ax = axes[0, 1]
        if methods and 'std_prediction' in uncertainty_results['methods'][methods[0]]:
            std_pred = uncertainty_results['methods'][methods[0]]['std_prediction']
            ax.hist(std_pred[sample_idx], bins=20, alpha=0.7)
            ax.set_title('Uncertainty Distribution (First Method)')
            ax.set_xlabel('Uncertainty')
            ax.set_ylabel('Frequency')
        
        # Plot 3: Confidence intervals
        ax = axes[1, 0]
        if 'confidence_intervals' in uncertainty_results['summary']:
            method_name = list(uncertainty_results['summary']['confidence_intervals'].keys())[0]
            ci_data = uncertainty_results['summary']['confidence_intervals'][method_name]
            
            mean_pred = uncertainty_results['methods'][method_name]['mean_prediction'][sample_idx]
            lower = ci_data['lower'][sample_idx]
            upper = ci_data['upper'][sample_idx]
            
            x_pos = range(len(mean_pred))
            ax.errorbar(x_pos, mean_pred, yerr=[mean_pred - lower, upper - mean_pred], 
                       fmt='o', capsize=5)
            ax.set_title('Prediction Confidence Intervals')
            ax.set_xlabel('MoA Class')
            ax.set_ylabel('Prediction')
        
        # Plot 4: Entropy vs Uncertainty
        ax = axes[1, 1]
        if methods and 'predictive_entropy' in uncertainty_results['methods'][methods[0]]:
            entropy = uncertainty_results['methods'][methods[0]]['predictive_entropy'][sample_idx]
            std_pred = uncertainty_results['methods'][methods[0]]['std_prediction'][sample_idx]
            
            ax.scatter(np.mean(std_pred), entropy, alpha=0.6)
            ax.set_xlabel('Mean Uncertainty')
            ax.set_ylabel('Predictive Entropy')
            ax.set_title('Entropy vs Uncertainty')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Uncertainty visualization saved to {output_path}")
        
        return fig
    
    def visualize_calibration(
        self,
        calibration_results: Dict[str, Any],
        class_idx: int = 0,
        output_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Visualize calibration results.
        
        Args:
            calibration_results: Results from calibrate_predictions
            class_idx: Class index to visualize
            output_path: Path to save the visualization
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Calibration curve (would need original data)
        ax = axes[0]
        ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
        ax.set_xlabel('Mean Predicted Probability')
        ax.set_ylabel('Fraction of Positives')
        ax.set_title(f'Calibration Curve - Class {class_idx}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Calibration metrics
        ax = axes[1]
        if class_idx in calibration_results['calibration_metrics']:
            metrics = calibration_results['calibration_metrics'][class_idx]
            
            metric_names = ['ECE Original', 'ECE Calibrated', 'MCE Original', 'MCE Calibrated']
            metric_values = [
                metrics.get('ece_original', 0),
                metrics.get('ece_calibrated', 0),
                metrics.get('mce_original', 0),
                metrics.get('mce_calibrated', 0)
            ]
            
            colors = ['red', 'blue', 'red', 'blue']
            bars = ax.bar(metric_names, metric_values, color=colors, alpha=0.7)
            ax.set_title(f'Calibration Metrics - Class {class_idx}')
            ax.set_ylabel('Error')
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Calibration visualization saved to {output_path}")
        
        return fig
