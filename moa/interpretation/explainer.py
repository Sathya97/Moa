"""
Main explainer class for MoA prediction models.

This module provides a unified interface for model interpretation,
combining attention visualization, counterfactual analysis, and feature importance.
"""

from typing import Dict, List, Tuple, Optional, Union, Any
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from torch_geometric.data import Data, Batch

from moa.utils.config import Config
from moa.utils.logger import get_logger
from .attention_viz import AttentionVisualizer
from .counterfactual import CounterfactualAnalyzer
from .feature_importance import FeatureImportanceAnalyzer

logger = get_logger(__name__)


class MoAExplainer:
    """
    Unified explainer for MoA prediction models.
    
    Provides comprehensive interpretation capabilities including:
    - Attention visualization for graph and pathway transformers
    - Counterfactual analysis for molecular fragments
    - Feature importance scoring across modalities
    - Prediction confidence analysis
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Config,
        moa_classes: List[str],
        output_dir: str = "interpretation_results"
    ):
        """
        Initialize MoA explainer.
        
        Args:
            model: Trained MoA prediction model
            config: Configuration object
            moa_classes: List of MoA class names
            output_dir: Directory for saving interpretation results
        """
        self.model = model
        self.config = config
        self.moa_classes = moa_classes
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Initialize interpretation components
        self.attention_viz = AttentionVisualizer(model, config)
        self.counterfactual_analyzer = CounterfactualAnalyzer(model, config)
        self.feature_importance = FeatureImportanceAnalyzer(model, config)
        
        logger.info(f"MoA Explainer initialized with {len(moa_classes)} classes")
    
    def explain_prediction(
        self,
        batch_data: Dict[str, torch.Tensor],
        sample_idx: int = 0,
        top_k_moas: int = 5,
        save_plots: bool = True
    ) -> Dict[str, Any]:
        """
        Provide comprehensive explanation for a single prediction.
        
        Args:
            batch_data: Input batch data
            sample_idx: Index of sample to explain within batch
            top_k_moas: Number of top predicted MoAs to explain
            save_plots: Whether to save visualization plots
            
        Returns:
            Dictionary containing explanation results
        """
        logger.info(f"Explaining prediction for sample {sample_idx}")
        
        with torch.no_grad():
            # Get model predictions and attention weights
            outputs = self.model(batch_data, return_attention=True)
            predictions = torch.sigmoid(outputs['logits'])
            
            # Extract single sample
            sample_predictions = predictions[sample_idx].cpu().numpy()
            
            # Get top predicted MoAs
            top_moa_indices = np.argsort(sample_predictions)[-top_k_moas:][::-1]
            top_moa_names = [self.moa_classes[i] for i in top_moa_indices]
            top_moa_scores = sample_predictions[top_moa_indices]
            
            logger.info(f"Top {top_k_moas} predicted MoAs: {top_moa_names}")
        
        explanation = {
            'sample_idx': sample_idx,
            'top_moas': {
                'names': top_moa_names,
                'indices': top_moa_indices.tolist(),
                'scores': top_moa_scores.tolist()
            },
            'attention_analysis': {},
            'counterfactual_analysis': {},
            'feature_importance': {},
            'modality_contributions': {}
        }
        
        # 1. Attention Analysis
        if 'attention_weights' in outputs:
            explanation['attention_analysis'] = self._analyze_attention(
                outputs['attention_weights'], batch_data, sample_idx, top_moa_indices
            )
        
        # 2. Counterfactual Analysis
        explanation['counterfactual_analysis'] = self._analyze_counterfactuals(
            batch_data, sample_idx, top_moa_indices
        )
        
        # 3. Feature Importance Analysis
        explanation['feature_importance'] = self._analyze_feature_importance(
            batch_data, sample_idx, top_moa_indices
        )
        
        # 4. Modality Contribution Analysis
        explanation['modality_contributions'] = self._analyze_modality_contributions(
            batch_data, sample_idx, top_moa_indices
        )
        
        # 5. Generate visualizations
        if save_plots:
            self._generate_explanation_plots(explanation, batch_data, sample_idx)
        
        return explanation
    
    def _analyze_attention(
        self,
        attention_weights: Dict[str, torch.Tensor],
        batch_data: Dict[str, torch.Tensor],
        sample_idx: int,
        top_moa_indices: np.ndarray
    ) -> Dict[str, Any]:
        """Analyze attention patterns for molecular and biological features."""
        attention_analysis = {}
        
        # Graph attention analysis
        if 'graph_attention' in attention_weights:
            graph_attention = attention_weights['graph_attention'][sample_idx]
            
            # Get molecular graph for this sample
            molecular_graph = self._extract_sample_graph(batch_data['molecular_graphs'], sample_idx)
            
            attention_analysis['molecular_attention'] = {
                'node_attention': graph_attention.cpu().numpy(),
                'important_nodes': torch.topk(graph_attention, k=min(10, len(graph_attention))).indices.cpu().numpy(),
                'attention_entropy': self._compute_attention_entropy(graph_attention)
            }
        
        # Pathway attention analysis
        if 'pathway_attention' in attention_weights:
            pathway_attention = attention_weights['pathway_attention'][sample_idx]
            
            attention_analysis['pathway_attention'] = {
                'feature_attention': pathway_attention.cpu().numpy(),
                'important_features': torch.topk(pathway_attention, k=min(20, len(pathway_attention))).indices.cpu().numpy(),
                'attention_entropy': self._compute_attention_entropy(pathway_attention)
            }
        
        return attention_analysis
    
    def _analyze_counterfactuals(
        self,
        batch_data: Dict[str, torch.Tensor],
        sample_idx: int,
        top_moa_indices: np.ndarray
    ) -> Dict[str, Any]:
        """Analyze counterfactual explanations for molecular fragments."""
        # Extract single sample data
        sample_data = self._extract_sample_data(batch_data, sample_idx)
        
        counterfactual_results = {}
        
        for moa_idx in top_moa_indices:
            moa_name = self.moa_classes[moa_idx]
            
            # Analyze molecular fragment importance
            fragment_importance = self.counterfactual_analyzer.analyze_fragment_importance(
                sample_data, target_moa=moa_idx
            )
            
            counterfactual_results[moa_name] = {
                'fragment_importance': fragment_importance,
                'critical_fragments': self._identify_critical_fragments(fragment_importance),
                'counterfactual_score': fragment_importance.get('overall_importance', 0.0)
            }
        
        return counterfactual_results
    
    def _analyze_feature_importance(
        self,
        batch_data: Dict[str, torch.Tensor],
        sample_idx: int,
        top_moa_indices: np.ndarray
    ) -> Dict[str, Any]:
        """Analyze feature importance across all modalities."""
        sample_data = self._extract_sample_data(batch_data, sample_idx)
        
        feature_importance = {}
        
        for moa_idx in top_moa_indices:
            moa_name = self.moa_classes[moa_idx]
            
            # Compute feature importance for each modality
            importance_scores = self.feature_importance.compute_feature_importance(
                sample_data, target_moa=moa_idx
            )
            
            feature_importance[moa_name] = importance_scores
        
        return feature_importance
    
    def _analyze_modality_contributions(
        self,
        batch_data: Dict[str, torch.Tensor],
        sample_idx: int,
        top_moa_indices: np.ndarray
    ) -> Dict[str, Any]:
        """Analyze contribution of each modality to predictions."""
        sample_data = self._extract_sample_data(batch_data, sample_idx)
        
        modality_contributions = {}
        
        # Get baseline prediction with all modalities
        with torch.no_grad():
            baseline_pred = self.model(batch_data)['logits'][sample_idx]
        
        # Test each modality individually
        modalities = ['molecular_graphs', 'mechtoken_features', 'gene_signature_features', 'pathway_score_features']
        
        for moa_idx in top_moa_indices:
            moa_name = self.moa_classes[moa_idx]
            baseline_score = torch.sigmoid(baseline_pred[moa_idx]).item()
            
            modality_scores = {}
            
            for modality in modalities:
                if modality in batch_data:
                    # Create modified batch with only this modality
                    modified_batch = self._create_single_modality_batch(batch_data, modality, sample_idx)
                    
                    with torch.no_grad():
                        modified_pred = self.model(modified_batch)['logits'][0]
                        modified_score = torch.sigmoid(modified_pred[moa_idx]).item()
                    
                    # Contribution is the difference from baseline
                    contribution = modified_score / baseline_score if baseline_score > 0 else 0.0
                    modality_scores[modality] = {
                        'score': modified_score,
                        'contribution': contribution,
                        'importance': abs(baseline_score - modified_score)
                    }
            
            modality_contributions[moa_name] = modality_scores
        
        return modality_contributions
    
    def _extract_sample_graph(self, batch_graphs: Batch, sample_idx: int) -> Data:
        """Extract individual graph from batch."""
        # This is a simplified extraction - in practice, you'd need to handle batch indexing
        return batch_graphs[sample_idx] if hasattr(batch_graphs, '__getitem__') else batch_graphs
    
    def _extract_sample_data(self, batch_data: Dict[str, torch.Tensor], sample_idx: int) -> Dict[str, torch.Tensor]:
        """Extract single sample from batch data."""
        sample_data = {}
        
        for key, value in batch_data.items():
            if key == 'molecular_graphs':
                sample_data[key] = self._extract_sample_graph(value, sample_idx)
            else:
                sample_data[key] = value[sample_idx:sample_idx+1]  # Keep batch dimension
        
        return sample_data
    
    def _create_single_modality_batch(
        self,
        batch_data: Dict[str, torch.Tensor],
        target_modality: str,
        sample_idx: int
    ) -> Dict[str, torch.Tensor]:
        """Create batch with only one modality active."""
        modified_batch = {}
        
        for key, value in batch_data.items():
            if key == target_modality:
                if key == 'molecular_graphs':
                    modified_batch[key] = self._extract_sample_graph(value, sample_idx)
                else:
                    modified_batch[key] = value[sample_idx:sample_idx+1]
            else:
                # Zero out other modalities
                if key == 'molecular_graphs':
                    # Create empty graph
                    empty_graph = Data(
                        x=torch.zeros(1, value.x.shape[1]),
                        edge_index=torch.zeros(2, 0, dtype=torch.long),
                        edge_attr=torch.zeros(0, value.edge_attr.shape[1]) if hasattr(value, 'edge_attr') else None
                    )
                    modified_batch[key] = empty_graph
                else:
                    modified_batch[key] = torch.zeros_like(value[sample_idx:sample_idx+1])
        
        return modified_batch
    
    def _compute_attention_entropy(self, attention_weights: torch.Tensor) -> float:
        """Compute entropy of attention distribution."""
        # Normalize attention weights
        attention_probs = torch.softmax(attention_weights, dim=-1)
        
        # Compute entropy
        entropy = -torch.sum(attention_probs * torch.log(attention_probs + 1e-8))
        return entropy.item()
    
    def _identify_critical_fragments(self, fragment_importance: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify most critical molecular fragments."""
        if 'fragment_scores' not in fragment_importance:
            return []
        
        fragment_scores = fragment_importance['fragment_scores']
        
        # Sort fragments by importance
        sorted_fragments = sorted(
            fragment_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Return top 5 critical fragments
        critical_fragments = []
        for fragment_id, score in sorted_fragments[:5]:
            critical_fragments.append({
                'fragment_id': fragment_id,
                'importance_score': score,
                'fragment_type': 'molecular_substructure'
            })
        
        return critical_fragments
    
    def _generate_explanation_plots(
        self,
        explanation: Dict[str, Any],
        batch_data: Dict[str, torch.Tensor],
        sample_idx: int
    ) -> None:
        """Generate visualization plots for explanation."""
        # Create sample-specific output directory
        sample_dir = self.output_dir / f"sample_{sample_idx}"
        sample_dir.mkdir(exist_ok=True)
        
        # 1. Top MoAs bar plot
        self._plot_top_moas(explanation, sample_dir)
        
        # 2. Modality contributions
        self._plot_modality_contributions(explanation, sample_dir)
        
        # 3. Attention heatmaps
        if explanation['attention_analysis']:
            self._plot_attention_analysis(explanation, sample_dir)
        
        # 4. Feature importance plots
        self._plot_feature_importance(explanation, sample_dir)
        
        logger.info(f"Explanation plots saved to {sample_dir}")
    
    def _plot_top_moas(self, explanation: Dict[str, Any], output_dir: Path) -> None:
        """Plot top predicted MoAs."""
        top_moas = explanation['top_moas']
        
        plt.figure(figsize=(10, 6))
        plt.barh(top_moas['names'], top_moas['scores'])
        plt.xlabel('Prediction Score')
        plt.title('Top Predicted Mechanisms of Action')
        plt.tight_layout()
        plt.savefig(output_dir / 'top_moas.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_modality_contributions(self, explanation: Dict[str, Any], output_dir: Path) -> None:
        """Plot modality contributions for each MoA."""
        modality_contrib = explanation['modality_contributions']
        
        if not modality_contrib:
            return
        
        # Create contribution matrix
        moa_names = list(modality_contrib.keys())
        modality_names = list(next(iter(modality_contrib.values())).keys())
        
        contrib_matrix = np.zeros((len(moa_names), len(modality_names)))
        
        for i, moa in enumerate(moa_names):
            for j, modality in enumerate(modality_names):
                contrib_matrix[i, j] = modality_contrib[moa][modality]['contribution']
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(
            contrib_matrix,
            xticklabels=modality_names,
            yticklabels=moa_names,
            annot=True,
            fmt='.3f',
            cmap='RdYlBu_r'
        )
        plt.title('Modality Contributions to MoA Predictions')
        plt.xlabel('Modality')
        plt.ylabel('Mechanism of Action')
        plt.tight_layout()
        plt.savefig(output_dir / 'modality_contributions.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_attention_analysis(self, explanation: Dict[str, Any], output_dir: Path) -> None:
        """Plot attention analysis results."""
        attention_analysis = explanation['attention_analysis']
        
        # Plot molecular attention if available
        if 'molecular_attention' in attention_analysis:
            mol_attention = attention_analysis['molecular_attention']
            
            plt.figure(figsize=(10, 6))
            plt.bar(range(len(mol_attention['node_attention'])), mol_attention['node_attention'])
            plt.xlabel('Node Index')
            plt.ylabel('Attention Weight')
            plt.title('Molecular Graph Node Attention')
            plt.tight_layout()
            plt.savefig(output_dir / 'molecular_attention.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # Plot pathway attention if available
        if 'pathway_attention' in attention_analysis:
            pathway_attention = attention_analysis['pathway_attention']
            
            plt.figure(figsize=(12, 6))
            plt.bar(range(len(pathway_attention['feature_attention'])), pathway_attention['feature_attention'])
            plt.xlabel('Feature Index')
            plt.ylabel('Attention Weight')
            plt.title('Pathway Feature Attention')
            plt.tight_layout()
            plt.savefig(output_dir / 'pathway_attention.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def _plot_feature_importance(self, explanation: Dict[str, Any], output_dir: Path) -> None:
        """Plot feature importance analysis."""
        feature_importance = explanation['feature_importance']
        
        if not feature_importance:
            return
        
        # Plot feature importance for top MoA
        top_moa = list(feature_importance.keys())[0]
        importance_data = feature_importance[top_moa]
        
        if 'modality_importance' in importance_data:
            modalities = list(importance_data['modality_importance'].keys())
            importance_scores = list(importance_data['modality_importance'].values())
            
            plt.figure(figsize=(10, 6))
            plt.bar(modalities, importance_scores)
            plt.xlabel('Modality')
            plt.ylabel('Importance Score')
            plt.title(f'Feature Importance for {top_moa}')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(output_dir / 'feature_importance.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def explain_batch(
        self,
        batch_data: Dict[str, torch.Tensor],
        top_k_moas: int = 3,
        save_individual: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Explain predictions for an entire batch.
        
        Args:
            batch_data: Input batch data
            top_k_moas: Number of top MoAs to explain per sample
            save_individual: Whether to save individual explanations
            
        Returns:
            List of explanation dictionaries for each sample
        """
        batch_size = len(batch_data[list(batch_data.keys())[0]])
        explanations = []
        
        logger.info(f"Explaining batch of {batch_size} samples")
        
        for sample_idx in range(batch_size):
            explanation = self.explain_prediction(
                batch_data, sample_idx, top_k_moas, save_plots=save_individual
            )
            explanations.append(explanation)
        
        return explanations
