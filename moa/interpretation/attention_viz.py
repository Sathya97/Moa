"""
Attention visualization for MoA prediction models.

This module provides tools for visualizing attention patterns in
graph transformers and pathway transformers.
"""

from typing import Dict, List, Tuple, Optional, Union, Any
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import networkx as nx
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_networkx

from moa.utils.config import Config
from moa.utils.logger import get_logger

logger = get_logger(__name__)


class AttentionVisualizer:
    """
    Visualizer for attention patterns in MoA prediction models.
    
    Provides visualization capabilities for:
    - Graph transformer attention on molecular structures
    - Pathway transformer attention on biological features
    - Cross-modal attention patterns
    - Attention flow analysis
    """
    
    def __init__(self, model: nn.Module, config: Config):
        """
        Initialize attention visualizer.
        
        Args:
            model: MoA prediction model with attention mechanisms
            config: Configuration object
        """
        self.model = model
        self.config = config
        self.model.eval()
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("viridis")
        
        logger.info("Attention visualizer initialized")
    
    def visualize_molecular_attention(
        self,
        molecular_graph: Data,
        attention_weights: torch.Tensor,
        output_path: Optional[str] = None,
        node_labels: Optional[List[str]] = None,
        title: str = "Molecular Graph Attention"
    ) -> plt.Figure:
        """
        Visualize attention weights on molecular graph.
        
        Args:
            molecular_graph: Molecular graph data
            attention_weights: Attention weights for each node
            output_path: Path to save the visualization
            node_labels: Optional labels for nodes
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        # Convert to NetworkX graph
        G = to_networkx(molecular_graph, to_undirected=True)
        
        # Normalize attention weights
        attention_weights = attention_weights.cpu().numpy()
        attention_weights = (attention_weights - attention_weights.min()) / (attention_weights.max() - attention_weights.min() + 1e-8)
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        
        # Set up layout
        pos = nx.spring_layout(G, k=1, iterations=50)
        
        # Draw edges
        nx.draw_networkx_edges(
            G, pos, ax=ax,
            edge_color='lightgray',
            width=1.0,
            alpha=0.6
        )
        
        # Draw nodes with attention-based coloring
        node_colors = plt.cm.Reds(attention_weights)
        nx.draw_networkx_nodes(
            G, pos, ax=ax,
            node_color=node_colors,
            node_size=300 + 500 * attention_weights,  # Size based on attention
            alpha=0.8
        )
        
        # Add node labels if provided
        if node_labels:
            labels = {i: label for i, label in enumerate(node_labels)}
            nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=8)
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=plt.cm.Reds, norm=plt.Normalize(vmin=0, vmax=1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
        cbar.set_label('Attention Weight', rotation=270, labelpad=20)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.axis('off')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Molecular attention visualization saved to {output_path}")
        
        return fig
    
    def visualize_pathway_attention(
        self,
        feature_names: List[str],
        attention_weights: torch.Tensor,
        output_path: Optional[str] = None,
        title: str = "Pathway Feature Attention",
        top_k: int = 20
    ) -> plt.Figure:
        """
        Visualize attention weights for pathway features.
        
        Args:
            feature_names: Names of pathway features
            attention_weights: Attention weights for each feature
            output_path: Path to save the visualization
            title: Plot title
            top_k: Number of top features to display
            
        Returns:
            Matplotlib figure
        """
        attention_weights = attention_weights.cpu().numpy()
        
        # Get top-k features
        top_indices = np.argsort(attention_weights)[-top_k:][::-1]
        top_features = [feature_names[i] for i in top_indices]
        top_weights = attention_weights[top_indices]
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Create horizontal bar plot
        bars = ax.barh(range(len(top_features)), top_weights, color='skyblue')
        
        # Customize plot
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features)
        ax.set_xlabel('Attention Weight')
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Add value labels on bars
        for i, (bar, weight) in enumerate(zip(bars, top_weights)):
            ax.text(weight + 0.01 * max(top_weights), i, f'{weight:.3f}', 
                   va='center', fontsize=10)
        
        # Invert y-axis to show highest attention at top
        ax.invert_yaxis()
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Pathway attention visualization saved to {output_path}")
        
        return fig
    
    def visualize_attention_heatmap(
        self,
        attention_matrix: torch.Tensor,
        row_labels: List[str],
        col_labels: List[str],
        output_path: Optional[str] = None,
        title: str = "Attention Heatmap"
    ) -> plt.Figure:
        """
        Visualize attention as a heatmap.
        
        Args:
            attention_matrix: 2D attention matrix
            row_labels: Labels for rows
            col_labels: Labels for columns
            output_path: Path to save the visualization
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        attention_matrix = attention_matrix.cpu().numpy()
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        
        # Create heatmap
        sns.heatmap(
            attention_matrix,
            xticklabels=col_labels,
            yticklabels=row_labels,
            annot=True if attention_matrix.shape[0] <= 10 else False,
            fmt='.3f',
            cmap='YlOrRd',
            ax=ax,
            cbar_kws={'label': 'Attention Weight'}
        )
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Attention heatmap saved to {output_path}")
        
        return fig
    
    def visualize_multi_head_attention(
        self,
        attention_weights: torch.Tensor,
        head_names: Optional[List[str]] = None,
        output_path: Optional[str] = None,
        title: str = "Multi-Head Attention Patterns"
    ) -> plt.Figure:
        """
        Visualize multi-head attention patterns.
        
        Args:
            attention_weights: Attention weights [num_heads, seq_len, seq_len]
            head_names: Optional names for attention heads
            output_path: Path to save the visualization
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        attention_weights = attention_weights.cpu().numpy()
        num_heads = attention_weights.shape[0]
        
        if head_names is None:
            head_names = [f"Head {i+1}" for i in range(num_heads)]
        
        # Create subplots
        fig, axes = plt.subplots(2, (num_heads + 1) // 2, figsize=(15, 8))
        if num_heads == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for i in range(num_heads):
            ax = axes[i]
            
            # Plot attention heatmap for this head
            im = ax.imshow(attention_weights[i], cmap='YlOrRd', aspect='auto')
            ax.set_title(head_names[i], fontsize=12)
            ax.set_xlabel('Key Position')
            ax.set_ylabel('Query Position')
            
            # Add colorbar
            plt.colorbar(im, ax=ax, shrink=0.8)
        
        # Hide unused subplots
        for i in range(num_heads, len(axes)):
            axes[i].axis('off')
        
        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Multi-head attention visualization saved to {output_path}")
        
        return fig
    
    def visualize_attention_flow(
        self,
        attention_weights: Dict[str, torch.Tensor],
        layer_names: List[str],
        output_path: Optional[str] = None,
        title: str = "Attention Flow Across Layers"
    ) -> plt.Figure:
        """
        Visualize attention flow across transformer layers.
        
        Args:
            attention_weights: Dictionary of attention weights per layer
            layer_names: Names of transformer layers
            output_path: Path to save the visualization
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        # Compute attention statistics per layer
        layer_stats = {}
        for layer_name in layer_names:
            if layer_name in attention_weights:
                attn = attention_weights[layer_name].cpu().numpy()
                
                # Compute statistics
                layer_stats[layer_name] = {
                    'mean_attention': np.mean(attn),
                    'max_attention': np.max(attn),
                    'attention_entropy': self._compute_attention_entropy(attn),
                    'attention_sparsity': np.sum(attn < 0.1) / attn.size
                }
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        layers = list(layer_stats.keys())
        
        # Plot mean attention
        mean_attns = [layer_stats[layer]['mean_attention'] for layer in layers]
        axes[0, 0].plot(layers, mean_attns, 'o-', linewidth=2, markersize=8)
        axes[0, 0].set_title('Mean Attention per Layer')
        axes[0, 0].set_ylabel('Mean Attention')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Plot max attention
        max_attns = [layer_stats[layer]['max_attention'] for layer in layers]
        axes[0, 1].plot(layers, max_attns, 's-', linewidth=2, markersize=8, color='orange')
        axes[0, 1].set_title('Max Attention per Layer')
        axes[0, 1].set_ylabel('Max Attention')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Plot attention entropy
        entropies = [layer_stats[layer]['attention_entropy'] for layer in layers]
        axes[1, 0].plot(layers, entropies, '^-', linewidth=2, markersize=8, color='green')
        axes[1, 0].set_title('Attention Entropy per Layer')
        axes[1, 0].set_ylabel('Entropy')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Plot attention sparsity
        sparsities = [layer_stats[layer]['attention_sparsity'] for layer in layers]
        axes[1, 1].plot(layers, sparsities, 'd-', linewidth=2, markersize=8, color='red')
        axes[1, 1].set_title('Attention Sparsity per Layer')
        axes[1, 1].set_ylabel('Sparsity')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Attention flow visualization saved to {output_path}")
        
        return fig
    
    def visualize_cross_modal_attention(
        self,
        chemical_features: torch.Tensor,
        biological_features: torch.Tensor,
        cross_attention: torch.Tensor,
        output_path: Optional[str] = None,
        title: str = "Cross-Modal Attention"
    ) -> plt.Figure:
        """
        Visualize cross-modal attention between chemical and biological features.
        
        Args:
            chemical_features: Chemical feature representations
            biological_features: Biological feature representations
            cross_attention: Cross-attention weights [chem_dim, bio_dim]
            output_path: Path to save the visualization
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        cross_attention = cross_attention.cpu().numpy()
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Create heatmap
        im = ax.imshow(cross_attention, cmap='RdYlBu_r', aspect='auto')
        
        ax.set_xlabel('Biological Features')
        ax.set_ylabel('Chemical Features')
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Cross-Attention Weight', rotation=270, labelpad=20)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Cross-modal attention visualization saved to {output_path}")
        
        return fig
    
    def _compute_attention_entropy(self, attention_weights: np.ndarray) -> float:
        """Compute entropy of attention distribution."""
        # Flatten and normalize
        attn_flat = attention_weights.flatten()
        attn_probs = attn_flat / (np.sum(attn_flat) + 1e-8)
        
        # Compute entropy
        entropy = -np.sum(attn_probs * np.log(attn_probs + 1e-8))
        return entropy
    
    def create_attention_summary(
        self,
        batch_data: Dict[str, torch.Tensor],
        output_dir: str,
        sample_indices: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """
        Create comprehensive attention analysis summary.
        
        Args:
            batch_data: Input batch data
            output_dir: Directory to save visualizations
            sample_indices: Specific samples to analyze
            
        Returns:
            Dictionary containing attention analysis results
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if sample_indices is None:
            sample_indices = [0]  # Analyze first sample by default
        
        summary = {
            'sample_analyses': {},
            'batch_statistics': {},
            'attention_patterns': {}
        }
        
        # Get model outputs with attention
        with torch.no_grad():
            outputs = self.model(batch_data, return_attention=True)
        
        if 'attention_weights' not in outputs:
            logger.warning("Model does not return attention weights")
            return summary
        
        attention_weights = outputs['attention_weights']
        
        # Analyze each specified sample
        for sample_idx in sample_indices:
            sample_summary = self._analyze_sample_attention(
                batch_data, attention_weights, sample_idx, output_dir
            )
            summary['sample_analyses'][sample_idx] = sample_summary
        
        # Compute batch-level statistics
        summary['batch_statistics'] = self._compute_batch_attention_stats(attention_weights)
        
        logger.info(f"Attention analysis summary created in {output_dir}")
        return summary
    
    def _analyze_sample_attention(
        self,
        batch_data: Dict[str, torch.Tensor],
        attention_weights: Dict[str, torch.Tensor],
        sample_idx: int,
        output_dir: Path
    ) -> Dict[str, Any]:
        """Analyze attention patterns for a single sample."""
        sample_dir = output_dir / f"sample_{sample_idx}"
        sample_dir.mkdir(exist_ok=True)
        
        sample_analysis = {}
        
        # Molecular attention analysis
        if 'graph_attention' in attention_weights:
            graph_attn = attention_weights['graph_attention'][sample_idx]
            
            # Visualize molecular attention
            if 'molecular_graphs' in batch_data:
                mol_graph = batch_data['molecular_graphs']
                # Extract single graph from batch (simplified)
                
                fig = self.visualize_molecular_attention(
                    mol_graph,
                    graph_attn,
                    output_path=str(sample_dir / 'molecular_attention.png')
                )
                plt.close(fig)
            
            sample_analysis['molecular_attention'] = {
                'entropy': self._compute_attention_entropy(graph_attn.cpu().numpy()),
                'max_attention': torch.max(graph_attn).item(),
                'mean_attention': torch.mean(graph_attn).item()
            }
        
        # Pathway attention analysis
        if 'pathway_attention' in attention_weights:
            pathway_attn = attention_weights['pathway_attention'][sample_idx]
            
            # Create feature names (simplified)
            feature_names = [f"Feature_{i}" for i in range(len(pathway_attn))]
            
            fig = self.visualize_pathway_attention(
                feature_names,
                pathway_attn,
                output_path=str(sample_dir / 'pathway_attention.png')
            )
            plt.close(fig)
            
            sample_analysis['pathway_attention'] = {
                'entropy': self._compute_attention_entropy(pathway_attn.cpu().numpy()),
                'max_attention': torch.max(pathway_attn).item(),
                'mean_attention': torch.mean(pathway_attn).item()
            }
        
        return sample_analysis
    
    def _compute_batch_attention_stats(self, attention_weights: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Compute batch-level attention statistics."""
        batch_stats = {}
        
        for attn_name, attn_tensor in attention_weights.items():
            attn_numpy = attn_tensor.cpu().numpy()
            
            batch_stats[attn_name] = {
                'mean': np.mean(attn_numpy),
                'std': np.std(attn_numpy),
                'min': np.min(attn_numpy),
                'max': np.max(attn_numpy),
                'entropy': self._compute_attention_entropy(attn_numpy)
            }
        
        return batch_stats
