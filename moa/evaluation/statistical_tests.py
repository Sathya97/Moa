"""
Statistical significance testing for model comparisons.

This module provides statistical tests to determine if differences
between model performances are statistically significant.
"""

from typing import Dict, List, Tuple, Optional, Union
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import wilcoxon, ttest_rel, mannwhitneyu, chi2_contingency
from sklearn.metrics import roc_auc_score, average_precision_score
import matplotlib.pyplot as plt
import seaborn as sns

from moa.utils.logger import get_logger

logger = get_logger(__name__)


class StatisticalTests:
    """Statistical significance testing for model comparisons."""
    
    def __init__(self, alpha: float = 0.05):
        """
        Initialize statistical tests.
        
        Args:
            alpha: Significance level for hypothesis testing
        """
        self.alpha = alpha
        
    def compare_models(
        self,
        predictions1: np.ndarray,
        predictions2: np.ndarray,
        true_labels: np.ndarray,
        test_type: str = "wilcoxon",
        metric: str = "auroc"
    ) -> Dict[str, float]:
        """
        Compare two models using statistical tests.
        
        Args:
            predictions1: Predictions from first model
            predictions2: Predictions from second model
            true_labels: True labels
            test_type: Type of statistical test
            metric: Metric to compare
            
        Returns:
            Dictionary with test results
        """
        # Compute per-sample or per-class metrics
        if metric == "auroc":
            scores1 = self._compute_per_class_auroc(true_labels, predictions1)
            scores2 = self._compute_per_class_auroc(true_labels, predictions2)
        elif metric == "auprc":
            scores1 = self._compute_per_class_auprc(true_labels, predictions1)
            scores2 = self._compute_per_class_auprc(true_labels, predictions2)
        elif metric == "accuracy":
            scores1 = self._compute_per_sample_accuracy(true_labels, predictions1)
            scores2 = self._compute_per_sample_accuracy(true_labels, predictions2)
        else:
            raise ValueError(f"Unsupported metric: {metric}")
        
        # Perform statistical test
        if test_type == "wilcoxon":
            return self._wilcoxon_test(scores1, scores2)
        elif test_type == "ttest":
            return self._paired_ttest(scores1, scores2)
        elif test_type == "mannwhitney":
            return self._mannwhitney_test(scores1, scores2)
        elif test_type == "mcnemar":
            return self._mcnemar_test(true_labels, predictions1, predictions2)
        else:
            raise ValueError(f"Unsupported test type: {test_type}")
    
    def multiple_comparisons_correction(
        self,
        p_values: List[float],
        method: str = "bonferroni"
    ) -> Tuple[List[float], List[bool]]:
        """
        Apply multiple comparisons correction.
        
        Args:
            p_values: List of p-values
            method: Correction method
            
        Returns:
            Tuple of (corrected_p_values, rejected_hypotheses)
        """
        p_values = np.array(p_values)
        
        if method == "bonferroni":
            corrected_p = p_values * len(p_values)
            corrected_p = np.minimum(corrected_p, 1.0)
            rejected = corrected_p < self.alpha
        elif method == "holm":
            return self._holm_correction(p_values)
        elif method == "fdr_bh":
            return self._benjamini_hochberg_correction(p_values)
        else:
            raise ValueError(f"Unsupported correction method: {method}")
        
        return corrected_p.tolist(), rejected.tolist()
    
    def bootstrap_confidence_interval(
        self,
        true_labels: np.ndarray,
        predictions: np.ndarray,
        metric_func: callable,
        n_bootstrap: int = 1000,
        confidence_level: float = 0.95
    ) -> Tuple[float, float, float]:
        """
        Compute bootstrap confidence interval for a metric.
        
        Args:
            true_labels: True labels
            predictions: Model predictions
            metric_func: Function to compute metric
            n_bootstrap: Number of bootstrap samples
            confidence_level: Confidence level
            
        Returns:
            Tuple of (metric_value, lower_bound, upper_bound)
        """
        n_samples = len(true_labels)
        bootstrap_scores = []
        
        # Original metric value
        original_score = metric_func(true_labels, predictions)
        
        # Bootstrap sampling
        for _ in range(n_bootstrap):
            # Sample with replacement
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            bootstrap_true = true_labels[indices]
            bootstrap_pred = predictions[indices]
            
            try:
                score = metric_func(bootstrap_true, bootstrap_pred)
                bootstrap_scores.append(score)
            except:
                # Skip invalid bootstrap samples
                continue
        
        bootstrap_scores = np.array(bootstrap_scores)
        
        # Compute confidence interval
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        lower_bound = np.percentile(bootstrap_scores, lower_percentile)
        upper_bound = np.percentile(bootstrap_scores, upper_percentile)
        
        return original_score, lower_bound, upper_bound
    
    def permutation_test(
        self,
        true_labels: np.ndarray,
        predictions1: np.ndarray,
        predictions2: np.ndarray,
        metric_func: callable,
        n_permutations: int = 1000
    ) -> Dict[str, float]:
        """
        Perform permutation test for model comparison.
        
        Args:
            true_labels: True labels
            predictions1: Predictions from first model
            predictions2: Predictions from second model
            metric_func: Function to compute metric
            n_permutations: Number of permutations
            
        Returns:
            Dictionary with test results
        """
        # Compute original difference
        score1 = metric_func(true_labels, predictions1)
        score2 = metric_func(true_labels, predictions2)
        original_diff = score1 - score2
        
        # Permutation test
        permuted_diffs = []
        combined_predictions = np.stack([predictions1, predictions2], axis=-1)
        
        for _ in range(n_permutations):
            # Randomly assign predictions to models
            permuted_assignments = np.random.randint(0, 2, size=len(true_labels))
            
            perm_pred1 = []
            perm_pred2 = []
            
            for i, assignment in enumerate(permuted_assignments):
                if assignment == 0:
                    perm_pred1.append(predictions1[i])
                    perm_pred2.append(predictions2[i])
                else:
                    perm_pred1.append(predictions2[i])
                    perm_pred2.append(predictions1[i])
            
            perm_pred1 = np.array(perm_pred1)
            perm_pred2 = np.array(perm_pred2)
            
            try:
                perm_score1 = metric_func(true_labels, perm_pred1)
                perm_score2 = metric_func(true_labels, perm_pred2)
                permuted_diffs.append(perm_score1 - perm_score2)
            except:
                continue
        
        permuted_diffs = np.array(permuted_diffs)
        
        # Compute p-value
        p_value = np.mean(np.abs(permuted_diffs) >= np.abs(original_diff))
        
        return {
            'original_difference': original_diff,
            'p_value': p_value,
            'significant': p_value < self.alpha,
            'n_permutations': len(permuted_diffs)
        }
    
    def _compute_per_class_auroc(self, true_labels: np.ndarray, predictions: np.ndarray) -> np.ndarray:
        """Compute AUROC for each class."""
        auroc_scores = []
        for i in range(true_labels.shape[1]):
            try:
                auroc = roc_auc_score(true_labels[:, i], predictions[:, i])
                auroc_scores.append(auroc)
            except:
                auroc_scores.append(0.5)  # Default for invalid cases
        
        return np.array(auroc_scores)
    
    def _compute_per_class_auprc(self, true_labels: np.ndarray, predictions: np.ndarray) -> np.ndarray:
        """Compute AUPRC for each class."""
        auprc_scores = []
        for i in range(true_labels.shape[1]):
            try:
                auprc = average_precision_score(true_labels[:, i], predictions[:, i])
                auprc_scores.append(auprc)
            except:
                auprc_scores.append(0.0)  # Default for invalid cases
        
        return np.array(auprc_scores)
    
    def _compute_per_sample_accuracy(self, true_labels: np.ndarray, predictions: np.ndarray) -> np.ndarray:
        """Compute accuracy for each sample."""
        # Convert probabilities to binary predictions
        binary_predictions = (predictions > 0.5).astype(int)
        
        # Compute per-sample accuracy (exact match)
        accuracies = []
        for i in range(len(true_labels)):
            accuracy = np.mean(true_labels[i] == binary_predictions[i])
            accuracies.append(accuracy)
        
        return np.array(accuracies)
    
    def _wilcoxon_test(self, scores1: np.ndarray, scores2: np.ndarray) -> Dict[str, float]:
        """Perform Wilcoxon signed-rank test."""
        try:
            statistic, p_value = wilcoxon(scores1, scores2, alternative='two-sided')
            
            return {
                'test_type': 'wilcoxon',
                'statistic': statistic,
                'p_value': p_value,
                'significant': p_value < self.alpha,
                'effect_size': self._compute_effect_size(scores1, scores2)
            }
        except Exception as e:
            logger.warning(f"Wilcoxon test failed: {e}")
            return {
                'test_type': 'wilcoxon',
                'statistic': np.nan,
                'p_value': 1.0,
                'significant': False,
                'effect_size': 0.0
            }
    
    def _paired_ttest(self, scores1: np.ndarray, scores2: np.ndarray) -> Dict[str, float]:
        """Perform paired t-test."""
        try:
            statistic, p_value = ttest_rel(scores1, scores2)
            
            return {
                'test_type': 'paired_ttest',
                'statistic': statistic,
                'p_value': p_value,
                'significant': p_value < self.alpha,
                'effect_size': self._compute_cohens_d(scores1, scores2)
            }
        except Exception as e:
            logger.warning(f"Paired t-test failed: {e}")
            return {
                'test_type': 'paired_ttest',
                'statistic': np.nan,
                'p_value': 1.0,
                'significant': False,
                'effect_size': 0.0
            }
    
    def _mannwhitney_test(self, scores1: np.ndarray, scores2: np.ndarray) -> Dict[str, float]:
        """Perform Mann-Whitney U test."""
        try:
            statistic, p_value = mannwhitneyu(scores1, scores2, alternative='two-sided')
            
            return {
                'test_type': 'mannwhitney',
                'statistic': statistic,
                'p_value': p_value,
                'significant': p_value < self.alpha,
                'effect_size': self._compute_effect_size(scores1, scores2)
            }
        except Exception as e:
            logger.warning(f"Mann-Whitney test failed: {e}")
            return {
                'test_type': 'mannwhitney',
                'statistic': np.nan,
                'p_value': 1.0,
                'significant': False,
                'effect_size': 0.0
            }
    
    def _mcnemar_test(
        self,
        true_labels: np.ndarray,
        predictions1: np.ndarray,
        predictions2: np.ndarray
    ) -> Dict[str, float]:
        """Perform McNemar's test."""
        try:
            # Convert to binary predictions
            binary_pred1 = (predictions1 > 0.5).astype(int)
            binary_pred2 = (predictions2 > 0.5).astype(int)
            
            # Compute contingency table for each class and aggregate
            total_b = 0  # Model 1 correct, Model 2 incorrect
            total_c = 0  # Model 1 incorrect, Model 2 correct
            
            for i in range(true_labels.shape[1]):
                correct1 = (binary_pred1[:, i] == true_labels[:, i])
                correct2 = (binary_pred2[:, i] == true_labels[:, i])
                
                b = np.sum(correct1 & ~correct2)
                c = np.sum(~correct1 & correct2)
                
                total_b += b
                total_c += c
            
            # McNemar's test statistic
            if total_b + total_c > 0:
                statistic = (abs(total_b - total_c) - 1) ** 2 / (total_b + total_c)
                p_value = 1 - stats.chi2.cdf(statistic, df=1)
            else:
                statistic = 0
                p_value = 1.0
            
            return {
                'test_type': 'mcnemar',
                'statistic': statistic,
                'p_value': p_value,
                'significant': p_value < self.alpha,
                'b_count': total_b,
                'c_count': total_c
            }
        except Exception as e:
            logger.warning(f"McNemar test failed: {e}")
            return {
                'test_type': 'mcnemar',
                'statistic': np.nan,
                'p_value': 1.0,
                'significant': False,
                'b_count': 0,
                'c_count': 0
            }
    
    def _compute_effect_size(self, scores1: np.ndarray, scores2: np.ndarray) -> float:
        """Compute effect size (rank-biserial correlation for non-parametric tests)."""
        try:
            # Rank-biserial correlation
            n1, n2 = len(scores1), len(scores2)
            U, _ = mannwhitneyu(scores1, scores2, alternative='two-sided')
            r = 1 - (2 * U) / (n1 * n2)
            return r
        except:
            return 0.0
    
    def _compute_cohens_d(self, scores1: np.ndarray, scores2: np.ndarray) -> float:
        """Compute Cohen's d effect size."""
        try:
            diff = scores1 - scores2
            pooled_std = np.sqrt((np.var(scores1) + np.var(scores2)) / 2)
            if pooled_std > 0:
                return np.mean(diff) / pooled_std
            else:
                return 0.0
        except:
            return 0.0
    
    def _holm_correction(self, p_values: np.ndarray) -> Tuple[List[float], List[bool]]:
        """Apply Holm correction for multiple comparisons."""
        p_values = np.array(p_values)
        n = len(p_values)
        
        # Sort p-values and keep track of original indices
        sorted_indices = np.argsort(p_values)
        sorted_p = p_values[sorted_indices]
        
        # Apply Holm correction
        corrected_p = np.zeros_like(p_values)
        rejected = np.zeros_like(p_values, dtype=bool)
        
        for i, (idx, p) in enumerate(zip(sorted_indices, sorted_p)):
            corrected_p[idx] = min(p * (n - i), 1.0)
            rejected[idx] = corrected_p[idx] < self.alpha
            
            # Stop if we fail to reject (Holm procedure)
            if not rejected[idx]:
                break
        
        return corrected_p.tolist(), rejected.tolist()
    
    def _benjamini_hochberg_correction(self, p_values: np.ndarray) -> Tuple[List[float], List[bool]]:
        """Apply Benjamini-Hochberg FDR correction."""
        p_values = np.array(p_values)
        n = len(p_values)
        
        # Sort p-values and keep track of original indices
        sorted_indices = np.argsort(p_values)
        sorted_p = p_values[sorted_indices]
        
        # Apply BH correction
        corrected_p = np.zeros_like(p_values)
        rejected = np.zeros_like(p_values, dtype=bool)
        
        for i, (idx, p) in enumerate(zip(sorted_indices, sorted_p)):
            corrected_p[idx] = min(p * n / (i + 1), 1.0)
        
        # Determine rejections (work backwards)
        for i in range(n - 1, -1, -1):
            idx = sorted_indices[i]
            if corrected_p[idx] < self.alpha:
                # Reject this and all previous hypotheses
                for j in range(i + 1):
                    rejected[sorted_indices[j]] = True
                break
        
        return corrected_p.tolist(), rejected.tolist()
