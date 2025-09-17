"""
Comprehensive evaluation metrics for MoA prediction.

This module implements multi-label classification metrics specifically
designed for mechanism of action prediction tasks.
"""

from typing import Dict, List, Tuple, Optional, Union
import warnings

import numpy as np
import torch
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score, accuracy_score,
    hamming_loss, jaccard_score, multilabel_confusion_matrix,
    classification_report
)
from sklearn.preprocessing import label_binarize
import pandas as pd

from moa.utils.logger import get_logger

logger = get_logger(__name__)


class MultiLabelMetrics:
    """Comprehensive multi-label classification metrics."""
    
    def __init__(self, num_classes: int, class_names: Optional[List[str]] = None):
        """
        Initialize multi-label metrics.
        
        Args:
            num_classes: Number of classes
            class_names: Optional list of class names
        """
        self.num_classes = num_classes
        self.class_names = class_names or [f"Class_{i}" for i in range(num_classes)]
        
    def compute_all_metrics(
        self,
        y_true: Union[np.ndarray, torch.Tensor],
        y_pred: Union[np.ndarray, torch.Tensor],
        y_pred_proba: Optional[Union[np.ndarray, torch.Tensor]] = None,
        threshold: float = 0.5
    ) -> Dict[str, float]:
        """
        Compute all multi-label metrics.
        
        Args:
            y_true: True binary labels (n_samples, n_classes)
            y_pred: Predicted binary labels (n_samples, n_classes)
            y_pred_proba: Predicted probabilities (n_samples, n_classes)
            threshold: Threshold for converting probabilities to binary predictions
            
        Returns:
            Dictionary of computed metrics
        """
        # Convert to numpy arrays
        y_true = self._to_numpy(y_true)
        y_pred = self._to_numpy(y_pred)
        
        if y_pred_proba is not None:
            y_pred_proba = self._to_numpy(y_pred_proba)
            # Convert probabilities to binary predictions if needed
            if y_pred is None:
                y_pred = (y_pred_proba > threshold).astype(int)
        
        metrics = {}
        
        # Basic classification metrics
        metrics.update(self._compute_classification_metrics(y_true, y_pred))
        
        # Ranking metrics (require probabilities)
        if y_pred_proba is not None:
            metrics.update(self._compute_ranking_metrics(y_true, y_pred_proba))
        
        # Multi-label specific metrics
        metrics.update(self._compute_multilabel_metrics(y_true, y_pred))
        
        # Per-class metrics
        per_class_metrics = self._compute_per_class_metrics(y_true, y_pred, y_pred_proba)
        metrics.update(per_class_metrics)
        
        return metrics
    
    def _compute_classification_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Compute basic classification metrics."""
        metrics = {}
        
        try:
            # Precision, Recall, F1 (macro and micro averages)
            metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
            metrics['precision_micro'] = precision_score(y_true, y_pred, average='micro', zero_division=0)
            metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
            metrics['recall_micro'] = recall_score(y_true, y_pred, average='micro', zero_division=0)
            metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
            metrics['f1_micro'] = f1_score(y_true, y_pred, average='micro', zero_division=0)
            
            # Weighted averages
            metrics['precision_weighted'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            metrics['recall_weighted'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
            metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            
        except Exception as e:
            logger.warning(f"Error computing classification metrics: {e}")
            # Set default values
            for metric in ['precision_macro', 'precision_micro', 'recall_macro', 'recall_micro', 
                          'f1_macro', 'f1_micro', 'precision_weighted', 'recall_weighted', 'f1_weighted']:
                metrics[metric] = 0.0
        
        return metrics
    
    def _compute_ranking_metrics(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray
    ) -> Dict[str, float]:
        """Compute ranking-based metrics."""
        metrics = {}
        
        try:
            # AUROC scores
            metrics['auroc_macro'] = roc_auc_score(y_true, y_pred_proba, average='macro')
            metrics['auroc_micro'] = roc_auc_score(y_true, y_pred_proba, average='micro')
            metrics['auroc_weighted'] = roc_auc_score(y_true, y_pred_proba, average='weighted')
            
            # AUPRC scores
            metrics['auprc_macro'] = average_precision_score(y_true, y_pred_proba, average='macro')
            metrics['auprc_micro'] = average_precision_score(y_true, y_pred_proba, average='micro')
            metrics['auprc_weighted'] = average_precision_score(y_true, y_pred_proba, average='weighted')
            
        except Exception as e:
            logger.warning(f"Error computing ranking metrics: {e}")
            # Set default values
            for metric in ['auroc_macro', 'auroc_micro', 'auroc_weighted', 
                          'auprc_macro', 'auprc_micro', 'auprc_weighted']:
                metrics[metric] = 0.5 if 'auroc' in metric else 0.0
        
        return metrics
    
    def _compute_multilabel_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Compute multi-label specific metrics."""
        metrics = {}
        
        try:
            # Subset accuracy (exact match)
            metrics['subset_accuracy'] = accuracy_score(y_true, y_pred)
            
            # Hamming loss
            metrics['hamming_loss'] = hamming_loss(y_true, y_pred)
            
            # Jaccard similarity
            metrics['jaccard_macro'] = jaccard_score(y_true, y_pred, average='macro', zero_division=0)
            metrics['jaccard_micro'] = jaccard_score(y_true, y_pred, average='micro', zero_division=0)
            metrics['jaccard_weighted'] = jaccard_score(y_true, y_pred, average='weighted', zero_division=0)
            
            # Label-based accuracy
            metrics['label_accuracy'] = self._compute_label_accuracy(y_true, y_pred)
            
            # Coverage error and ranking loss (if applicable)
            metrics.update(self._compute_coverage_metrics(y_true, y_pred))
            
        except Exception as e:
            logger.warning(f"Error computing multi-label metrics: {e}")
            # Set default values
            metrics.update({
                'subset_accuracy': 0.0,
                'hamming_loss': 1.0,
                'jaccard_macro': 0.0,
                'jaccard_micro': 0.0,
                'jaccard_weighted': 0.0,
                'label_accuracy': 0.0
            })
        
        return metrics
    
    def _compute_per_class_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """Compute per-class metrics."""
        metrics = {}
        
        try:
            # Per-class precision, recall, F1
            precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
            recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
            f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
            
            # Store per-class metrics
            for i, class_name in enumerate(self.class_names):
                metrics[f'precision_{class_name}'] = precision_per_class[i]
                metrics[f'recall_{class_name}'] = recall_per_class[i]
                metrics[f'f1_{class_name}'] = f1_per_class[i]
            
            # Per-class AUROC and AUPRC (if probabilities available)
            if y_pred_proba is not None:
                for i, class_name in enumerate(self.class_names):
                    try:
                        auroc = roc_auc_score(y_true[:, i], y_pred_proba[:, i])
                        auprc = average_precision_score(y_true[:, i], y_pred_proba[:, i])
                        metrics[f'auroc_{class_name}'] = auroc
                        metrics[f'auprc_{class_name}'] = auprc
                    except:
                        metrics[f'auroc_{class_name}'] = 0.5
                        metrics[f'auprc_{class_name}'] = 0.0
            
        except Exception as e:
            logger.warning(f"Error computing per-class metrics: {e}")
        
        return metrics
    
    def _compute_label_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute label-based accuracy."""
        # Accuracy for each label independently
        label_accuracies = []
        for i in range(y_true.shape[1]):
            acc = accuracy_score(y_true[:, i], y_pred[:, i])
            label_accuracies.append(acc)
        
        return np.mean(label_accuracies)
    
    def _compute_coverage_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Compute coverage-related metrics."""
        metrics = {}
        
        try:
            # Label cardinality (average number of labels per sample)
            true_cardinality = np.mean(np.sum(y_true, axis=1))
            pred_cardinality = np.mean(np.sum(y_pred, axis=1))
            
            metrics['true_label_cardinality'] = true_cardinality
            metrics['pred_label_cardinality'] = pred_cardinality
            metrics['cardinality_difference'] = abs(true_cardinality - pred_cardinality)
            
            # Label density
            metrics['label_density'] = true_cardinality / y_true.shape[1]
            
        except Exception as e:
            logger.warning(f"Error computing coverage metrics: {e}")
            metrics.update({
                'true_label_cardinality': 0.0,
                'pred_label_cardinality': 0.0,
                'cardinality_difference': 0.0,
                'label_density': 0.0
            })
        
        return metrics
    
    def _to_numpy(self, tensor: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """Convert tensor to numpy array."""
        if isinstance(tensor, torch.Tensor):
            return tensor.detach().cpu().numpy()
        return tensor
    
    def get_classification_report(
        self,
        y_true: Union[np.ndarray, torch.Tensor],
        y_pred: Union[np.ndarray, torch.Tensor]
    ) -> str:
        """Get detailed classification report."""
        y_true = self._to_numpy(y_true)
        y_pred = self._to_numpy(y_pred)
        
        try:
            report = classification_report(
                y_true, y_pred,
                target_names=self.class_names,
                zero_division=0
            )
            return report
        except Exception as e:
            logger.warning(f"Error generating classification report: {e}")
            return "Classification report generation failed"
    
    def get_confusion_matrices(
        self,
        y_true: Union[np.ndarray, torch.Tensor],
        y_pred: Union[np.ndarray, torch.Tensor]
    ) -> Dict[str, np.ndarray]:
        """Get confusion matrices for each class."""
        y_true = self._to_numpy(y_true)
        y_pred = self._to_numpy(y_pred)
        
        try:
            # Multi-label confusion matrices
            cm_matrices = multilabel_confusion_matrix(y_true, y_pred)
            
            confusion_matrices = {}
            for i, class_name in enumerate(self.class_names):
                confusion_matrices[class_name] = cm_matrices[i]
            
            return confusion_matrices
        except Exception as e:
            logger.warning(f"Error computing confusion matrices: {e}")
            return {}


class MoAMetrics(MultiLabelMetrics):
    """Specialized metrics for MoA prediction."""
    
    def __init__(self, moa_classes: List[str]):
        """
        Initialize MoA-specific metrics.
        
        Args:
            moa_classes: List of MoA class names
        """
        super().__init__(len(moa_classes), moa_classes)
        self.moa_classes = moa_classes
    
    def compute_moa_specific_metrics(
        self,
        y_true: Union[np.ndarray, torch.Tensor],
        y_pred: Union[np.ndarray, torch.Tensor],
        y_pred_proba: Optional[Union[np.ndarray, torch.Tensor]] = None,
        compound_ids: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Compute MoA-specific evaluation metrics.
        
        Args:
            y_true: True MoA labels
            y_pred: Predicted MoA labels
            y_pred_proba: Predicted probabilities
            compound_ids: Optional compound identifiers
            
        Returns:
            Dictionary of MoA-specific metrics
        """
        # Get standard multi-label metrics
        metrics = self.compute_all_metrics(y_true, y_pred, y_pred_proba)
        
        # Add MoA-specific metrics
        y_true = self._to_numpy(y_true)
        y_pred = self._to_numpy(y_pred)
        
        # Top-k accuracy metrics
        if y_pred_proba is not None:
            y_pred_proba = self._to_numpy(y_pred_proba)
            metrics.update(self._compute_topk_metrics(y_true, y_pred_proba))
        
        # Pathway-level metrics (if pathway information available)
        metrics.update(self._compute_pathway_metrics(y_true, y_pred))
        
        # Drug discovery specific metrics
        metrics.update(self._compute_drug_discovery_metrics(y_true, y_pred, y_pred_proba))
        
        return metrics
    
    def _compute_topk_metrics(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        k_values: List[int] = [1, 3, 5, 10]
    ) -> Dict[str, float]:
        """Compute top-k accuracy metrics."""
        metrics = {}
        
        for k in k_values:
            if k > y_true.shape[1]:
                continue
            
            # Top-k accuracy: at least one true label in top-k predictions
            top_k_acc = 0.0
            for i in range(y_true.shape[0]):
                true_labels = np.where(y_true[i] == 1)[0]
                if len(true_labels) > 0:
                    top_k_preds = np.argsort(y_pred_proba[i])[-k:]
                    if len(np.intersect1d(true_labels, top_k_preds)) > 0:
                        top_k_acc += 1.0
            
            metrics[f'top_{k}_accuracy'] = top_k_acc / y_true.shape[0]
        
        return metrics
    
    def _compute_pathway_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Compute pathway-level evaluation metrics."""
        metrics = {}
        
        # This would require pathway hierarchy information
        # For now, implement basic pathway grouping metrics
        
        # Group MoAs by pathway (simplified example)
        pathway_groups = self._get_pathway_groups()
        
        for pathway_name, moa_indices in pathway_groups.items():
            if len(moa_indices) > 0:
                # Pathway-level precision/recall
                pathway_true = y_true[:, moa_indices]
                pathway_pred = y_pred[:, moa_indices]
                
                # Any prediction in pathway
                pathway_true_any = np.any(pathway_true, axis=1)
                pathway_pred_any = np.any(pathway_pred, axis=1)
                
                if np.sum(pathway_true_any) > 0:
                    pathway_precision = np.sum(pathway_true_any & pathway_pred_any) / max(np.sum(pathway_pred_any), 1)
                    pathway_recall = np.sum(pathway_true_any & pathway_pred_any) / np.sum(pathway_true_any)
                    
                    metrics[f'pathway_{pathway_name}_precision'] = pathway_precision
                    metrics[f'pathway_{pathway_name}_recall'] = pathway_recall
        
        return metrics
    
    def _compute_drug_discovery_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """Compute drug discovery specific metrics."""
        metrics = {}
        
        # Hit rate (percentage of compounds with at least one correct prediction)
        hit_rate = 0.0
        for i in range(y_true.shape[0]):
            true_labels = np.where(y_true[i] == 1)[0]
            pred_labels = np.where(y_pred[i] == 1)[0]
            if len(true_labels) > 0 and len(np.intersect1d(true_labels, pred_labels)) > 0:
                hit_rate += 1.0
        
        metrics['hit_rate'] = hit_rate / y_true.shape[0]
        
        # False discovery rate
        total_predictions = np.sum(y_pred)
        false_predictions = np.sum(y_pred & (1 - y_true))
        metrics['false_discovery_rate'] = false_predictions / max(total_predictions, 1)
        
        # Enrichment metrics (if probabilities available)
        if y_pred_proba is not None:
            metrics.update(self._compute_enrichment_metrics(y_true, y_pred_proba))
        
        return metrics
    
    def _compute_enrichment_metrics(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        percentiles: List[float] = [1, 5, 10, 20]
    ) -> Dict[str, float]:
        """Compute enrichment metrics for drug discovery."""
        metrics = {}
        
        for percentile in percentiles:
            # Top percentile enrichment
            threshold = np.percentile(y_pred_proba.flatten(), 100 - percentile)
            top_predictions = y_pred_proba >= threshold
            
            # Enrichment factor
            total_positives = np.sum(y_true)
            top_positives = np.sum(y_true & top_predictions)
            expected_positives = total_positives * (percentile / 100)
            
            if expected_positives > 0:
                enrichment = top_positives / expected_positives
                metrics[f'enrichment_top_{percentile}%'] = enrichment
        
        return metrics
    
    def _get_pathway_groups(self) -> Dict[str, List[int]]:
        """Get pathway groupings for MoA classes."""
        # This is a simplified example - in practice, this would come from
        # biological pathway databases like Reactome or KEGG
        
        pathway_groups = {
            'cell_cycle': [],
            'apoptosis': [],
            'metabolism': [],
            'signaling': [],
            'transport': [],
            'other': []
        }
        
        # Simple keyword-based grouping
        for i, moa_class in enumerate(self.moa_classes):
            moa_lower = moa_class.lower()
            
            if any(keyword in moa_lower for keyword in ['cycle', 'mitosis', 'division']):
                pathway_groups['cell_cycle'].append(i)
            elif any(keyword in moa_lower for keyword in ['apoptosis', 'death', 'necrosis']):
                pathway_groups['apoptosis'].append(i)
            elif any(keyword in moa_lower for keyword in ['metabolism', 'metabolic', 'synthesis']):
                pathway_groups['metabolism'].append(i)
            elif any(keyword in moa_lower for keyword in ['signaling', 'signal', 'pathway']):
                pathway_groups['signaling'].append(i)
            elif any(keyword in moa_lower for keyword in ['transport', 'channel', 'pump']):
                pathway_groups['transport'].append(i)
            else:
                pathway_groups['other'].append(i)
        
        return pathway_groups

    def get_moa_summary_report(
        self,
        y_true: Union[np.ndarray, torch.Tensor],
        y_pred: Union[np.ndarray, torch.Tensor],
        y_pred_proba: Optional[Union[np.ndarray, torch.Tensor]] = None
    ) -> pd.DataFrame:
        """
        Generate comprehensive MoA evaluation summary.

        Args:
            y_true: True MoA labels
            y_pred: Predicted MoA labels
            y_pred_proba: Predicted probabilities

        Returns:
            DataFrame with per-MoA performance metrics
        """
        y_true = self._to_numpy(y_true)
        y_pred = self._to_numpy(y_pred)

        # Compute per-class metrics
        precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)

        # Support (number of true instances per class)
        support_per_class = np.sum(y_true, axis=0)

        # Create summary DataFrame
        summary_data = {
            'MoA_Class': self.moa_classes,
            'Precision': precision_per_class,
            'Recall': recall_per_class,
            'F1_Score': f1_per_class,
            'Support': support_per_class
        }

        # Add AUROC and AUPRC if probabilities available
        if y_pred_proba is not None:
            y_pred_proba = self._to_numpy(y_pred_proba)
            auroc_per_class = []
            auprc_per_class = []

            for i in range(len(self.moa_classes)):
                try:
                    auroc = roc_auc_score(y_true[:, i], y_pred_proba[:, i])
                    auprc = average_precision_score(y_true[:, i], y_pred_proba[:, i])
                except:
                    auroc = 0.5
                    auprc = 0.0

                auroc_per_class.append(auroc)
                auprc_per_class.append(auprc)

            summary_data['AUROC'] = auroc_per_class
            summary_data['AUPRC'] = auprc_per_class

        summary_df = pd.DataFrame(summary_data)

        # Sort by F1 score descending
        summary_df = summary_df.sort_values('F1_Score', ascending=False)

        return summary_df
