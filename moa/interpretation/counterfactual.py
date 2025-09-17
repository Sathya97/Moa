"""
Counterfactual analysis for molecular fragments and biological features.

This module implements counterfactual reasoning to identify causal
molecular substructures and biological features for MoA predictions.
"""

from typing import Dict, List, Tuple, Optional, Union, Any
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from copy import deepcopy
from torch_geometric.data import Data, Batch

from moa.utils.config import Config
from moa.utils.logger import get_logger

logger = get_logger(__name__)


class CounterfactualAnalyzer:
    """
    Counterfactual analyzer for MoA prediction models.
    
    Implements counterfactual reasoning to identify:
    - Causal molecular fragments for chemical predictions
    - Important biological features for pathway predictions
    - Cross-modal dependencies between modalities
    """
    
    def __init__(self, model: nn.Module, config: Config):
        """
        Initialize counterfactual analyzer.
        
        Args:
            model: MoA prediction model
            config: Configuration object
        """
        self.model = model
        self.config = config
        self.model.eval()
        
        # Counterfactual analysis parameters
        self.perturbation_strength = config.get('interpretation.counterfactual.perturbation_strength', 0.1)
        self.num_perturbations = config.get('interpretation.counterfactual.num_perturbations', 100)
        self.fragment_size_range = config.get('interpretation.counterfactual.fragment_size_range', (2, 8))
        
        logger.info("Counterfactual analyzer initialized")
    
    def analyze_fragment_importance(
        self,
        sample_data: Dict[str, torch.Tensor],
        target_moa: int,
        method: str = 'node_removal'
    ) -> Dict[str, Any]:
        """
        Analyze importance of molecular fragments using counterfactual reasoning.
        
        Args:
            sample_data: Single sample data
            target_moa: Target MoA index to analyze
            method: Counterfactual method ('node_removal', 'edge_removal', 'feature_masking')
            
        Returns:
            Dictionary containing fragment importance analysis
        """
        logger.info(f"Analyzing fragment importance for MoA {target_moa} using {method}")
        
        # Get baseline prediction
        with torch.no_grad():
            baseline_output = self.model(sample_data)
            baseline_pred = torch.sigmoid(baseline_output['logits'])[0, target_moa].item()
        
        fragment_analysis = {
            'baseline_prediction': baseline_pred,
            'method': method,
            'fragment_scores': {},
            'overall_importance': 0.0,
            'critical_fragments': []
        }
        
        if method == 'node_removal':
            fragment_analysis.update(self._analyze_node_removal(sample_data, target_moa, baseline_pred))
        elif method == 'edge_removal':
            fragment_analysis.update(self._analyze_edge_removal(sample_data, target_moa, baseline_pred))
        elif method == 'feature_masking':
            fragment_analysis.update(self._analyze_feature_masking(sample_data, target_moa, baseline_pred))
        else:
            raise ValueError(f"Unknown counterfactual method: {method}")
        
        return fragment_analysis
    
    def _analyze_node_removal(
        self,
        sample_data: Dict[str, torch.Tensor],
        target_moa: int,
        baseline_pred: float
    ) -> Dict[str, Any]:
        """Analyze fragment importance by removing individual nodes."""
        molecular_graph = sample_data['molecular_graphs']
        num_nodes = molecular_graph.x.shape[0]
        
        node_importance = {}
        
        for node_idx in range(num_nodes):
            # Create modified graph without this node
            modified_graph = self._remove_node(molecular_graph, node_idx)
            
            # Create modified sample data
            modified_data = deepcopy(sample_data)
            modified_data['molecular_graphs'] = modified_graph
            
            # Get prediction without this node
            with torch.no_grad():
                modified_output = self.model(modified_data)
                modified_pred = torch.sigmoid(modified_output['logits'])[0, target_moa].item()
            
            # Compute importance as prediction change
            importance = abs(baseline_pred - modified_pred)
            node_importance[f'node_{node_idx}'] = importance
        
        # Identify critical nodes
        sorted_nodes = sorted(node_importance.items(), key=lambda x: x[1], reverse=True)
        critical_nodes = sorted_nodes[:min(5, len(sorted_nodes))]
        
        return {
            'fragment_scores': node_importance,
            'overall_importance': np.mean(list(node_importance.values())),
            'critical_fragments': [{'fragment_id': node_id, 'importance': score} 
                                 for node_id, score in critical_nodes]
        }
    
    def _analyze_edge_removal(
        self,
        sample_data: Dict[str, torch.Tensor],
        target_moa: int,
        baseline_pred: float
    ) -> Dict[str, Any]:
        """Analyze fragment importance by removing individual edges."""
        molecular_graph = sample_data['molecular_graphs']
        edge_index = molecular_graph.edge_index
        num_edges = edge_index.shape[1]
        
        edge_importance = {}
        
        for edge_idx in range(num_edges):
            # Create modified graph without this edge
            modified_graph = self._remove_edge(molecular_graph, edge_idx)
            
            # Create modified sample data
            modified_data = deepcopy(sample_data)
            modified_data['molecular_graphs'] = modified_graph
            
            # Get prediction without this edge
            with torch.no_grad():
                modified_output = self.model(modified_data)
                modified_pred = torch.sigmoid(modified_output['logits'])[0, target_moa].item()
            
            # Compute importance as prediction change
            importance = abs(baseline_pred - modified_pred)
            edge_importance[f'edge_{edge_idx}'] = importance
        
        # Identify critical edges
        sorted_edges = sorted(edge_importance.items(), key=lambda x: x[1], reverse=True)
        critical_edges = sorted_edges[:min(5, len(sorted_edges))]
        
        return {
            'fragment_scores': edge_importance,
            'overall_importance': np.mean(list(edge_importance.values())),
            'critical_fragments': [{'fragment_id': edge_id, 'importance': score} 
                                 for edge_id, score in critical_edges]
        }
    
    def _analyze_feature_masking(
        self,
        sample_data: Dict[str, torch.Tensor],
        target_moa: int,
        baseline_pred: float
    ) -> Dict[str, Any]:
        """Analyze importance by masking node features."""
        molecular_graph = sample_data['molecular_graphs']
        num_nodes, num_features = molecular_graph.x.shape
        
        feature_importance = {}
        
        # Test masking each feature dimension
        for feat_idx in range(num_features):
            # Create modified graph with masked feature
            modified_graph = deepcopy(molecular_graph)
            modified_graph.x[:, feat_idx] = 0  # Mask feature
            
            # Create modified sample data
            modified_data = deepcopy(sample_data)
            modified_data['molecular_graphs'] = modified_graph
            
            # Get prediction with masked feature
            with torch.no_grad():
                modified_output = self.model(modified_data)
                modified_pred = torch.sigmoid(modified_output['logits'])[0, target_moa].item()
            
            # Compute importance as prediction change
            importance = abs(baseline_pred - modified_pred)
            feature_importance[f'feature_{feat_idx}'] = importance
        
        # Test masking each node's features
        for node_idx in range(num_nodes):
            # Create modified graph with masked node features
            modified_graph = deepcopy(molecular_graph)
            modified_graph.x[node_idx, :] = 0  # Mask all features for this node
            
            # Create modified sample data
            modified_data = deepcopy(sample_data)
            modified_data['molecular_graphs'] = modified_graph
            
            # Get prediction with masked node
            with torch.no_grad():
                modified_output = self.model(modified_data)
                modified_pred = torch.sigmoid(modified_output['logits'])[0, target_moa].item()
            
            # Compute importance as prediction change
            importance = abs(baseline_pred - modified_pred)
            feature_importance[f'node_features_{node_idx}'] = importance
        
        # Identify critical features
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        critical_features = sorted_features[:min(10, len(sorted_features))]
        
        return {
            'fragment_scores': feature_importance,
            'overall_importance': np.mean(list(feature_importance.values())),
            'critical_fragments': [{'fragment_id': feat_id, 'importance': score} 
                                 for feat_id, score in critical_features]
        }
    
    def _remove_node(self, graph: Data, node_idx: int) -> Data:
        """Remove a node from the molecular graph."""
        num_nodes = graph.x.shape[0]
        
        if num_nodes <= 1:
            # Cannot remove the last node
            return graph
        
        # Create node mask
        node_mask = torch.ones(num_nodes, dtype=torch.bool)
        node_mask[node_idx] = False
        
        # Filter nodes
        new_x = graph.x[node_mask]
        
        # Filter edges
        edge_mask = (graph.edge_index[0] != node_idx) & (graph.edge_index[1] != node_idx)
        new_edge_index = graph.edge_index[:, edge_mask]
        
        # Adjust edge indices (shift down indices > node_idx)
        new_edge_index[new_edge_index > node_idx] -= 1
        
        # Filter edge attributes if present
        new_edge_attr = None
        if hasattr(graph, 'edge_attr') and graph.edge_attr is not None:
            new_edge_attr = graph.edge_attr[edge_mask]
        
        # Create new graph
        new_graph = Data(
            x=new_x,
            edge_index=new_edge_index,
            edge_attr=new_edge_attr
        )
        
        return new_graph
    
    def _remove_edge(self, graph: Data, edge_idx: int) -> Data:
        """Remove an edge from the molecular graph."""
        num_edges = graph.edge_index.shape[1]
        
        if num_edges <= 0:
            return graph
        
        # Create edge mask
        edge_mask = torch.ones(num_edges, dtype=torch.bool)
        edge_mask[edge_idx] = False
        
        # Filter edges
        new_edge_index = graph.edge_index[:, edge_mask]
        
        # Filter edge attributes if present
        new_edge_attr = None
        if hasattr(graph, 'edge_attr') and graph.edge_attr is not None:
            new_edge_attr = graph.edge_attr[edge_mask]
        
        # Create new graph
        new_graph = Data(
            x=graph.x,
            edge_index=new_edge_index,
            edge_attr=new_edge_attr
        )
        
        return new_graph
    
    def analyze_biological_feature_importance(
        self,
        sample_data: Dict[str, torch.Tensor],
        target_moa: int,
        feature_type: str = 'mechtoken_features'
    ) -> Dict[str, Any]:
        """
        Analyze importance of biological features using counterfactual reasoning.
        
        Args:
            sample_data: Single sample data
            target_moa: Target MoA index to analyze
            feature_type: Type of biological features to analyze
            
        Returns:
            Dictionary containing biological feature importance analysis
        """
        if feature_type not in sample_data:
            logger.warning(f"Feature type {feature_type} not found in sample data")
            return {}
        
        # Get baseline prediction
        with torch.no_grad():
            baseline_output = self.model(sample_data)
            baseline_pred = torch.sigmoid(baseline_output['logits'])[0, target_moa].item()
        
        features = sample_data[feature_type]
        num_features = features.shape[1]
        
        feature_importance = {}
        
        # Test masking each feature
        for feat_idx in range(num_features):
            # Create modified data with masked feature
            modified_data = deepcopy(sample_data)
            modified_data[feature_type] = features.clone()
            modified_data[feature_type][0, feat_idx] = 0  # Mask feature
            
            # Get prediction with masked feature
            with torch.no_grad():
                modified_output = self.model(modified_data)
                modified_pred = torch.sigmoid(modified_output['logits'])[0, target_moa].item()
            
            # Compute importance as prediction change
            importance = abs(baseline_pred - modified_pred)
            feature_importance[f'{feature_type}_{feat_idx}'] = importance
        
        # Identify critical features
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        critical_features = sorted_features[:min(20, len(sorted_features))]
        
        return {
            'baseline_prediction': baseline_pred,
            'feature_type': feature_type,
            'feature_scores': feature_importance,
            'overall_importance': np.mean(list(feature_importance.values())),
            'critical_features': [{'feature_id': feat_id, 'importance': score} 
                                for feat_id, score in critical_features]
        }
    
    def analyze_modality_dependencies(
        self,
        sample_data: Dict[str, torch.Tensor],
        target_moa: int
    ) -> Dict[str, Any]:
        """
        Analyze dependencies between different modalities using counterfactual reasoning.
        
        Args:
            sample_data: Single sample data
            target_moa: Target MoA index to analyze
            
        Returns:
            Dictionary containing modality dependency analysis
        """
        # Get baseline prediction with all modalities
        with torch.no_grad():
            baseline_output = self.model(sample_data)
            baseline_pred = torch.sigmoid(baseline_output['logits'])[0, target_moa].item()
        
        modalities = ['molecular_graphs', 'mechtoken_features', 'gene_signature_features', 'pathway_score_features']
        available_modalities = [mod for mod in modalities if mod in sample_data]
        
        dependency_analysis = {
            'baseline_prediction': baseline_pred,
            'modality_contributions': {},
            'pairwise_dependencies': {},
            'modality_ranking': []
        }
        
        # Test each modality individually
        individual_contributions = {}
        for modality in available_modalities:
            # Create data with only this modality
            single_mod_data = self._create_single_modality_data(sample_data, modality)
            
            with torch.no_grad():
                single_output = self.model(single_mod_data)
                single_pred = torch.sigmoid(single_output['logits'])[0, target_moa].item()
            
            individual_contributions[modality] = single_pred
            dependency_analysis['modality_contributions'][modality] = {
                'individual_prediction': single_pred,
                'contribution_ratio': single_pred / baseline_pred if baseline_pred > 0 else 0
            }
        
        # Test pairwise modality combinations
        for i, mod1 in enumerate(available_modalities):
            for j, mod2 in enumerate(available_modalities[i+1:], i+1):
                # Create data with only these two modalities
                pair_data = self._create_pair_modality_data(sample_data, mod1, mod2)
                
                with torch.no_grad():
                    pair_output = self.model(pair_data)
                    pair_pred = torch.sigmoid(pair_output['logits'])[0, target_moa].item()
                
                # Compute synergy (interaction effect)
                expected_additive = individual_contributions[mod1] + individual_contributions[mod2]
                synergy = pair_pred - expected_additive
                
                dependency_analysis['pairwise_dependencies'][f'{mod1}_{mod2}'] = {
                    'pair_prediction': pair_pred,
                    'expected_additive': expected_additive,
                    'synergy': synergy,
                    'synergy_ratio': synergy / baseline_pred if baseline_pred > 0 else 0
                }
        
        # Rank modalities by importance
        modality_importance = [(mod, contrib['individual_prediction']) 
                             for mod, contrib in dependency_analysis['modality_contributions'].items()]
        dependency_analysis['modality_ranking'] = sorted(modality_importance, key=lambda x: x[1], reverse=True)
        
        return dependency_analysis
    
    def _create_single_modality_data(
        self,
        sample_data: Dict[str, torch.Tensor],
        target_modality: str
    ) -> Dict[str, torch.Tensor]:
        """Create sample data with only one modality active."""
        single_mod_data = {}
        
        for key, value in sample_data.items():
            if key == target_modality:
                single_mod_data[key] = value
            else:
                # Zero out other modalities
                if key == 'molecular_graphs':
                    # Create minimal graph
                    empty_graph = Data(
                        x=torch.zeros(1, value.x.shape[1]),
                        edge_index=torch.zeros(2, 0, dtype=torch.long),
                        edge_attr=torch.zeros(0, value.edge_attr.shape[1]) if hasattr(value, 'edge_attr') else None
                    )
                    single_mod_data[key] = empty_graph
                else:
                    single_mod_data[key] = torch.zeros_like(value)
        
        return single_mod_data
    
    def _create_pair_modality_data(
        self,
        sample_data: Dict[str, torch.Tensor],
        modality1: str,
        modality2: str
    ) -> Dict[str, torch.Tensor]:
        """Create sample data with only two modalities active."""
        pair_data = {}
        
        for key, value in sample_data.items():
            if key in [modality1, modality2]:
                pair_data[key] = value
            else:
                # Zero out other modalities
                if key == 'molecular_graphs':
                    # Create minimal graph
                    empty_graph = Data(
                        x=torch.zeros(1, value.x.shape[1]),
                        edge_index=torch.zeros(2, 0, dtype=torch.long),
                        edge_attr=torch.zeros(0, value.edge_attr.shape[1]) if hasattr(value, 'edge_attr') else None
                    )
                    pair_data[key] = empty_graph
                else:
                    pair_data[key] = torch.zeros_like(value)
        
        return pair_data
    
    def generate_counterfactual_explanations(
        self,
        sample_data: Dict[str, torch.Tensor],
        target_moa: int,
        num_explanations: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Generate multiple counterfactual explanations for a prediction.
        
        Args:
            sample_data: Single sample data
            target_moa: Target MoA index to analyze
            num_explanations: Number of explanations to generate
            
        Returns:
            List of counterfactual explanations
        """
        explanations = []
        
        # 1. Node removal explanations
        node_analysis = self.analyze_fragment_importance(sample_data, target_moa, 'node_removal')
        explanations.append({
            'type': 'molecular_node_removal',
            'analysis': node_analysis,
            'description': 'Importance of molecular graph nodes'
        })
        
        # 2. Edge removal explanations
        edge_analysis = self.analyze_fragment_importance(sample_data, target_moa, 'edge_removal')
        explanations.append({
            'type': 'molecular_edge_removal',
            'analysis': edge_analysis,
            'description': 'Importance of molecular graph edges'
        })
        
        # 3. Biological feature explanations
        for feature_type in ['mechtoken_features', 'gene_signature_features', 'pathway_score_features']:
            if feature_type in sample_data:
                bio_analysis = self.analyze_biological_feature_importance(sample_data, target_moa, feature_type)
                explanations.append({
                    'type': f'biological_{feature_type}',
                    'analysis': bio_analysis,
                    'description': f'Importance of {feature_type}'
                })
        
        # 4. Modality dependency explanations
        dependency_analysis = self.analyze_modality_dependencies(sample_data, target_moa)
        explanations.append({
            'type': 'modality_dependencies',
            'analysis': dependency_analysis,
            'description': 'Dependencies between different data modalities'
        })
        
        return explanations[:num_explanations]
