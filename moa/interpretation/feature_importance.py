"""
Feature importance analysis for MoA prediction models.

This module implements various feature importance methods including
gradient-based attribution, integrated gradients, and SHAP-like explanations.
"""

from typing import Dict, List, Tuple, Optional, Union, Any
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from copy import deepcopy
from torch_geometric.data import Data

from moa.utils.config import Config
from moa.utils.logger import get_logger

logger = get_logger(__name__)


class FeatureImportanceAnalyzer:
    """
    Feature importance analyzer for MoA prediction models.
    
    Implements multiple attribution methods:
    - Gradient-based attribution
    - Integrated gradients
    - Layer-wise relevance propagation (LRP)
    - Permutation importance
    - SHAP-like explanations
    """
    
    def __init__(self, model: nn.Module, config: Config):
        """
        Initialize feature importance analyzer.
        
        Args:
            model: MoA prediction model
            config: Configuration object
        """
        self.model = model
        self.config = config
        
        # Feature importance parameters
        self.num_integrated_steps = config.get('interpretation.feature_importance.integrated_steps', 50)
        self.num_permutations = config.get('interpretation.feature_importance.num_permutations', 100)
        self.baseline_strategy = config.get('interpretation.feature_importance.baseline_strategy', 'zero')
        
        logger.info("Feature importance analyzer initialized")
    
    def compute_feature_importance(
        self,
        sample_data: Dict[str, torch.Tensor],
        target_moa: int,
        methods: List[str] = ['gradient', 'integrated_gradients', 'permutation']
    ) -> Dict[str, Any]:
        """
        Compute feature importance using multiple methods.
        
        Args:
            sample_data: Single sample data
            target_moa: Target MoA index to analyze
            methods: List of attribution methods to use
            
        Returns:
            Dictionary containing feature importance results
        """
        logger.info(f"Computing feature importance for MoA {target_moa} using methods: {methods}")
        
        importance_results = {
            'target_moa': target_moa,
            'methods': {},
            'modality_importance': {},
            'summary': {}
        }
        
        # Compute importance for each method
        for method in methods:
            if method == 'gradient':
                importance_results['methods']['gradient'] = self._compute_gradient_attribution(
                    sample_data, target_moa
                )
            elif method == 'integrated_gradients':
                importance_results['methods']['integrated_gradients'] = self._compute_integrated_gradients(
                    sample_data, target_moa
                )
            elif method == 'permutation':
                importance_results['methods']['permutation'] = self._compute_permutation_importance(
                    sample_data, target_moa
                )
            elif method == 'lrp':
                importance_results['methods']['lrp'] = self._compute_lrp_attribution(
                    sample_data, target_moa
                )
            else:
                logger.warning(f"Unknown attribution method: {method}")
        
        # Compute modality-level importance
        importance_results['modality_importance'] = self._compute_modality_importance(
            sample_data, target_moa
        )
        
        # Create summary
        importance_results['summary'] = self._create_importance_summary(importance_results)
        
        return importance_results
    
    def _compute_gradient_attribution(
        self,
        sample_data: Dict[str, torch.Tensor],
        target_moa: int
    ) -> Dict[str, Any]:
        """Compute gradient-based feature attribution."""
        self.model.eval()
        
        # Prepare data with gradients
        grad_data = {}
        for key, value in sample_data.items():
            if key == 'molecular_graphs':
                # Enable gradients for molecular graph features
                grad_graph = deepcopy(value)
                grad_graph.x.requires_grad_(True)
                grad_data[key] = grad_graph
            else:
                # Enable gradients for other features
                grad_data[key] = value.clone().requires_grad_(True)
        
        # Forward pass
        output = self.model(grad_data)
        target_score = output['logits'][0, target_moa]
        
        # Backward pass
        target_score.backward()
        
        # Extract gradients
        attributions = {}
        
        for key, value in grad_data.items():
            if key == 'molecular_graphs':
                if value.x.grad is not None:
                    # Compute node-level attributions
                    node_attributions = torch.norm(value.x.grad, dim=1).detach().cpu().numpy()
                    attributions[f'{key}_nodes'] = node_attributions
                    
                    # Compute feature-level attributions
                    feature_attributions = torch.norm(value.x.grad, dim=0).detach().cpu().numpy()
                    attributions[f'{key}_features'] = feature_attributions
            else:
                if value.grad is not None:
                    # Compute feature attributions
                    feature_attributions = torch.abs(value.grad).squeeze().detach().cpu().numpy()
                    attributions[key] = feature_attributions
        
        return {
            'method': 'gradient',
            'attributions': attributions,
            'total_attribution': sum(np.sum(attr) for attr in attributions.values())
        }
    
    def _compute_integrated_gradients(
        self,
        sample_data: Dict[str, torch.Tensor],
        target_moa: int
    ) -> Dict[str, Any]:
        """Compute integrated gradients attribution."""
        self.model.eval()
        
        # Create baseline data
        baseline_data = self._create_baseline_data(sample_data)
        
        # Compute integrated gradients
        attributions = {}
        
        for key in sample_data.keys():
            if key == 'molecular_graphs':
                # Integrated gradients for molecular graphs
                mol_attributions = self._integrated_gradients_molecular(
                    sample_data[key], baseline_data[key], target_moa, sample_data
                )
                attributions.update(mol_attributions)
            else:
                # Integrated gradients for other features
                feature_attributions = self._integrated_gradients_features(
                    sample_data[key], baseline_data[key], target_moa, sample_data, key
                )
                attributions[key] = feature_attributions
        
        return {
            'method': 'integrated_gradients',
            'attributions': attributions,
            'num_steps': self.num_integrated_steps,
            'total_attribution': sum(np.sum(attr) for attr in attributions.values())
        }
    
    def _integrated_gradients_molecular(
        self,
        input_graph: Data,
        baseline_graph: Data,
        target_moa: int,
        full_sample_data: Dict[str, torch.Tensor]
    ) -> Dict[str, np.ndarray]:
        """Compute integrated gradients for molecular graph."""
        # Interpolate between baseline and input
        integrated_grads_nodes = torch.zeros_like(input_graph.x)
        integrated_grads_features = torch.zeros(input_graph.x.shape[1])
        
        for step in range(self.num_integrated_steps):
            alpha = step / (self.num_integrated_steps - 1)
            
            # Interpolated graph
            interp_graph = deepcopy(baseline_graph)
            interp_graph.x = baseline_graph.x + alpha * (input_graph.x - baseline_graph.x)
            interp_graph.x.requires_grad_(True)
            
            # Create interpolated sample data
            interp_data = deepcopy(full_sample_data)
            interp_data['molecular_graphs'] = interp_graph
            
            # Forward pass
            output = self.model(interp_data)
            target_score = output['logits'][0, target_moa]
            
            # Backward pass
            target_score.backward()
            
            # Accumulate gradients
            if interp_graph.x.grad is not None:
                integrated_grads_nodes += interp_graph.x.grad
                integrated_grads_features += torch.norm(interp_graph.x.grad, dim=0)
        
        # Average and multiply by input difference
        integrated_grads_nodes = integrated_grads_nodes / self.num_integrated_steps
        integrated_grads_features = integrated_grads_features / self.num_integrated_steps
        
        input_diff = input_graph.x - baseline_graph.x
        node_attributions = (integrated_grads_nodes * input_diff).sum(dim=1).detach().cpu().numpy()
        feature_attributions = (integrated_grads_features * torch.norm(input_diff, dim=0)).detach().cpu().numpy()
        
        return {
            'molecular_graphs_nodes': node_attributions,
            'molecular_graphs_features': feature_attributions
        }
    
    def _integrated_gradients_features(
        self,
        input_features: torch.Tensor,
        baseline_features: torch.Tensor,
        target_moa: int,
        full_sample_data: Dict[str, torch.Tensor],
        feature_key: str
    ) -> np.ndarray:
        """Compute integrated gradients for feature vectors."""
        integrated_grads = torch.zeros_like(input_features)
        
        for step in range(self.num_integrated_steps):
            alpha = step / (self.num_integrated_steps - 1)
            
            # Interpolated features
            interp_features = baseline_features + alpha * (input_features - baseline_features)
            interp_features.requires_grad_(True)
            
            # Create interpolated sample data
            interp_data = deepcopy(full_sample_data)
            interp_data[feature_key] = interp_features
            
            # Forward pass
            output = self.model(interp_data)
            target_score = output['logits'][0, target_moa]
            
            # Backward pass
            target_score.backward()
            
            # Accumulate gradients
            if interp_features.grad is not None:
                integrated_grads += interp_features.grad
        
        # Average and multiply by input difference
        integrated_grads = integrated_grads / self.num_integrated_steps
        input_diff = input_features - baseline_features
        attributions = (integrated_grads * input_diff).squeeze().detach().cpu().numpy()
        
        return attributions
    
    def _compute_permutation_importance(
        self,
        sample_data: Dict[str, torch.Tensor],
        target_moa: int
    ) -> Dict[str, Any]:
        """Compute permutation-based feature importance."""
        self.model.eval()
        
        # Get baseline prediction
        with torch.no_grad():
            baseline_output = self.model(sample_data)
            baseline_pred = torch.sigmoid(baseline_output['logits'])[0, target_moa].item()
        
        permutation_importance = {}
        
        for key, value in sample_data.items():
            if key == 'molecular_graphs':
                # Permutation importance for molecular graph
                mol_importance = self._permutation_importance_molecular(
                    sample_data, target_moa, baseline_pred
                )
                permutation_importance.update(mol_importance)
            else:
                # Permutation importance for feature vectors
                feature_importance = self._permutation_importance_features(
                    sample_data, target_moa, baseline_pred, key
                )
                permutation_importance[key] = feature_importance
        
        return {
            'method': 'permutation',
            'attributions': permutation_importance,
            'baseline_prediction': baseline_pred,
            'num_permutations': self.num_permutations
        }
    
    def _permutation_importance_molecular(
        self,
        sample_data: Dict[str, torch.Tensor],
        target_moa: int,
        baseline_pred: float
    ) -> Dict[str, np.ndarray]:
        """Compute permutation importance for molecular graph."""
        molecular_graph = sample_data['molecular_graphs']
        num_nodes = molecular_graph.x.shape[0]
        num_features = molecular_graph.x.shape[1]
        
        # Node-level permutation importance
        node_importance = np.zeros(num_nodes)
        
        for node_idx in range(num_nodes):
            importance_scores = []
            
            for _ in range(min(self.num_permutations, 20)):  # Limit for efficiency
                # Create permuted data
                permuted_data = deepcopy(sample_data)
                permuted_graph = deepcopy(molecular_graph)
                
                # Permute this node's features
                perm_indices = torch.randperm(num_features)
                permuted_graph.x[node_idx] = permuted_graph.x[node_idx][perm_indices]
                permuted_data['molecular_graphs'] = permuted_graph
                
                # Get prediction
                with torch.no_grad():
                    output = self.model(permuted_data)
                    pred = torch.sigmoid(output['logits'])[0, target_moa].item()
                
                # Compute importance as prediction change
                importance = abs(baseline_pred - pred)
                importance_scores.append(importance)
            
            node_importance[node_idx] = np.mean(importance_scores)
        
        # Feature-level permutation importance
        feature_importance = np.zeros(num_features)
        
        for feat_idx in range(num_features):
            importance_scores = []
            
            for _ in range(min(self.num_permutations, 20)):
                # Create permuted data
                permuted_data = deepcopy(sample_data)
                permuted_graph = deepcopy(molecular_graph)
                
                # Permute this feature across all nodes
                perm_indices = torch.randperm(num_nodes)
                permuted_graph.x[:, feat_idx] = permuted_graph.x[perm_indices, feat_idx]
                permuted_data['molecular_graphs'] = permuted_graph
                
                # Get prediction
                with torch.no_grad():
                    output = self.model(permuted_data)
                    pred = torch.sigmoid(output['logits'])[0, target_moa].item()
                
                # Compute importance
                importance = abs(baseline_pred - pred)
                importance_scores.append(importance)
            
            feature_importance[feat_idx] = np.mean(importance_scores)
        
        return {
            'molecular_graphs_nodes': node_importance,
            'molecular_graphs_features': feature_importance
        }
    
    def _permutation_importance_features(
        self,
        sample_data: Dict[str, torch.Tensor],
        target_moa: int,
        baseline_pred: float,
        feature_key: str
    ) -> np.ndarray:
        """Compute permutation importance for feature vectors."""
        features = sample_data[feature_key]
        num_features = features.shape[1]
        
        feature_importance = np.zeros(num_features)
        
        for feat_idx in range(num_features):
            importance_scores = []
            
            for _ in range(min(self.num_permutations, 50)):
                # Create permuted data
                permuted_data = deepcopy(sample_data)
                permuted_features = features.clone()
                
                # Permute this feature
                permuted_features[0, feat_idx] = torch.randn_like(permuted_features[0, feat_idx])
                permuted_data[feature_key] = permuted_features
                
                # Get prediction
                with torch.no_grad():
                    output = self.model(permuted_data)
                    pred = torch.sigmoid(output['logits'])[0, target_moa].item()
                
                # Compute importance
                importance = abs(baseline_pred - pred)
                importance_scores.append(importance)
            
            feature_importance[feat_idx] = np.mean(importance_scores)
        
        return feature_importance
    
    def _compute_lrp_attribution(
        self,
        sample_data: Dict[str, torch.Tensor],
        target_moa: int
    ) -> Dict[str, Any]:
        """Compute Layer-wise Relevance Propagation (LRP) attribution."""
        # Simplified LRP implementation
        # In practice, this would require model-specific LRP rules
        
        logger.warning("LRP attribution is simplified - full implementation requires model-specific rules")
        
        # Use gradient as approximation for now
        gradient_attr = self._compute_gradient_attribution(sample_data, target_moa)
        
        return {
            'method': 'lrp_simplified',
            'attributions': gradient_attr['attributions'],
            'note': 'Simplified LRP using gradients'
        }
    
    def _compute_modality_importance(
        self,
        sample_data: Dict[str, torch.Tensor],
        target_moa: int
    ) -> Dict[str, float]:
        """Compute importance of each modality."""
        # Get baseline prediction
        with torch.no_grad():
            baseline_output = self.model(sample_data)
            baseline_pred = torch.sigmoid(baseline_output['logits'])[0, target_moa].item()
        
        modality_importance = {}
        
        # Test removing each modality
        modalities = ['molecular_graphs', 'mechtoken_features', 'gene_signature_features', 'pathway_score_features']
        
        for modality in modalities:
            if modality in sample_data:
                # Create data without this modality
                modified_data = deepcopy(sample_data)
                
                if modality == 'molecular_graphs':
                    # Create minimal graph
                    empty_graph = Data(
                        x=torch.zeros(1, sample_data[modality].x.shape[1]),
                        edge_index=torch.zeros(2, 0, dtype=torch.long),
                        edge_attr=torch.zeros(0, sample_data[modality].edge_attr.shape[1]) 
                        if hasattr(sample_data[modality], 'edge_attr') else None
                    )
                    modified_data[modality] = empty_graph
                else:
                    modified_data[modality] = torch.zeros_like(sample_data[modality])
                
                # Get prediction without this modality
                with torch.no_grad():
                    output = self.model(modified_data)
                    pred = torch.sigmoid(output['logits'])[0, target_moa].item()
                
                # Importance is the prediction change
                importance = abs(baseline_pred - pred)
                modality_importance[modality] = importance
        
        return modality_importance
    
    def _create_baseline_data(self, sample_data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Create baseline data for integrated gradients."""
        baseline_data = {}
        
        for key, value in sample_data.items():
            if self.baseline_strategy == 'zero':
                if key == 'molecular_graphs':
                    # Zero baseline for molecular graph
                    baseline_graph = Data(
                        x=torch.zeros_like(value.x),
                        edge_index=value.edge_index,
                        edge_attr=torch.zeros_like(value.edge_attr) if hasattr(value, 'edge_attr') else None
                    )
                    baseline_data[key] = baseline_graph
                else:
                    baseline_data[key] = torch.zeros_like(value)
            elif self.baseline_strategy == 'mean':
                # Use mean values as baseline (would require dataset statistics)
                baseline_data[key] = torch.zeros_like(value)  # Simplified
            else:
                baseline_data[key] = torch.zeros_like(value)
        
        return baseline_data
    
    def _create_importance_summary(self, importance_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create summary of feature importance results."""
        summary = {
            'top_molecular_nodes': [],
            'top_molecular_features': [],
            'top_biological_features': {},
            'modality_ranking': []
        }
        
        # Aggregate results across methods
        all_attributions = {}
        
        for method_name, method_results in importance_results['methods'].items():
            for attr_name, attr_values in method_results['attributions'].items():
                if attr_name not in all_attributions:
                    all_attributions[attr_name] = []
                all_attributions[attr_name].append(attr_values)
        
        # Average attributions across methods
        averaged_attributions = {}
        for attr_name, attr_list in all_attributions.items():
            averaged_attributions[attr_name] = np.mean(attr_list, axis=0)
        
        # Extract top features
        for attr_name, attr_values in averaged_attributions.items():
            if 'molecular_graphs_nodes' in attr_name:
                top_indices = np.argsort(attr_values)[-5:][::-1]
                summary['top_molecular_nodes'] = [
                    {'node_idx': int(idx), 'importance': float(attr_values[idx])}
                    for idx in top_indices
                ]
            elif 'molecular_graphs_features' in attr_name:
                top_indices = np.argsort(attr_values)[-10:][::-1]
                summary['top_molecular_features'] = [
                    {'feature_idx': int(idx), 'importance': float(attr_values[idx])}
                    for idx in top_indices
                ]
            else:
                # Biological features
                top_indices = np.argsort(attr_values)[-10:][::-1]
                summary['top_biological_features'][attr_name] = [
                    {'feature_idx': int(idx), 'importance': float(attr_values[idx])}
                    for idx in top_indices
                ]
        
        # Rank modalities
        if importance_results['modality_importance']:
            modality_items = list(importance_results['modality_importance'].items())
            summary['modality_ranking'] = sorted(modality_items, key=lambda x: x[1], reverse=True)
        
        return summary
