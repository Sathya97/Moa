#!/usr/bin/env python3
"""
Model Architecture Demonstration for MoA prediction framework.

This script demonstrates the complete multi-modal deep learning architecture including:
- Graph Transformer for chemical features
- Pathway Transformer for biological features
- Hypergraph Neural Networks for multi-modal fusion
- Multi-objective loss functions
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch_geometric.data import Data, Batch

# Add the project root to the path
sys.path.append(str(Path(__file__).parent.parent))

from moa.utils.config import Config
from moa.models.multimodal_model import MultiModalMoAPredictor
from moa.models.graph_transformer import GraphTransformer
from moa.models.pathway_transformer import PathwayTransformer
from moa.models.hypergraph_layers import HypergraphFusion
from moa.models.losses import MultiObjectiveLoss


def create_sample_molecular_graphs(batch_size: int = 4) -> Batch:
    """Create sample molecular graphs for testing."""
    graphs = []
    
    for i in range(batch_size):
        # Create a random molecular graph
        num_nodes = np.random.randint(10, 30)
        num_edges = np.random.randint(num_nodes, num_nodes * 2)
        
        # Node features (atomic features)
        node_features = torch.randn(num_nodes, 64)  # 64-dim node features
        
        # Edge indices
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        
        # Edge features (bond features)
        edge_attr = torch.randn(num_edges, 16)  # 16-dim edge features
        
        graph = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)
        graphs.append(graph)
    
    return Batch.from_data_list(graphs)


def create_sample_biological_features(batch_size: int = 4) -> dict:
    """Create sample biological features for testing."""
    return {
        "mechtoken_features": torch.randn(batch_size, 128),  # Mechanism tokens
        "gene_signature_features": torch.randn(batch_size, 978),  # Gene signatures
        "pathway_score_features": torch.randn(batch_size, 50),  # Pathway scores
        "protein_pocket_features": torch.randn(batch_size, 256)  # Protein pockets
    }


def create_sample_targets(batch_size: int = 4, num_classes: int = 20) -> torch.Tensor:
    """Create sample multi-label targets."""
    # Create sparse multi-label targets
    targets = torch.zeros(batch_size, num_classes)
    for i in range(batch_size):
        # Each sample has 1-3 positive labels
        num_positive = np.random.randint(1, 4)
        positive_indices = np.random.choice(num_classes, num_positive, replace=False)
        targets[i, positive_indices] = 1.0
    
    return targets


def test_graph_transformer():
    """Test Graph Transformer component."""
    print("Testing Graph Transformer...")
    
    config = Config("configs/config.yaml")
    config.set("data.num_moa_classes", 20)
    
    # Create model
    graph_transformer = GraphTransformer(config)
    
    # Create sample data
    batch_size = 4
    molecular_graphs = create_sample_molecular_graphs(batch_size)
    
    # Forward pass
    with torch.no_grad():
        embeddings = graph_transformer(
            molecular_graphs.x,
            molecular_graphs.edge_index,
            molecular_graphs.edge_attr,
            molecular_graphs.batch
        )
    
    print(f"  Input: {molecular_graphs.x.shape[0]} nodes, {molecular_graphs.edge_index.shape[1]} edges")
    print(f"  Output: {embeddings.shape}")
    print(f"  ✓ Graph Transformer working correctly")
    
    return embeddings


def test_pathway_transformer():
    """Test Pathway Transformer component."""
    print("\nTesting Pathway Transformer...")
    
    config = Config("configs/config.yaml")
    config.set("data.num_moa_classes", 20)
    
    # Create model
    pathway_transformer = PathwayTransformer(config)
    
    # Create sample data
    batch_size = 4
    bio_features = create_sample_biological_features(batch_size)
    
    # Forward pass
    with torch.no_grad():
        embeddings = pathway_transformer(
            bio_features["mechtoken_features"],
            bio_features["gene_signature_features"],
            bio_features["pathway_score_features"]
        )
    
    print(f"  Input: MechTokens {bio_features['mechtoken_features'].shape}, "
          f"Genes {bio_features['gene_signature_features'].shape}, "
          f"Pathways {bio_features['pathway_score_features'].shape}")
    print(f"  Output: {embeddings.shape}")
    print(f"  ✓ Pathway Transformer working correctly")
    
    return embeddings


def test_hypergraph_fusion():
    """Test Hypergraph Fusion component."""
    print("\nTesting Hypergraph Fusion...")
    
    # Create sample modality features
    batch_size = 4
    modality_dims = {
        "chemistry": 256,
        "biology": 256,
        "structure": 256
    }
    
    modality_features = {
        modality: torch.randn(batch_size, dim)
        for modality, dim in modality_dims.items()
    }
    
    # Create fusion layer
    fusion_layer = HypergraphFusion(
        modality_dims=modality_dims,
        hidden_dim=256,
        num_hypergraph_layers=3,
        num_attention_heads=8
    )
    
    # Forward pass
    with torch.no_grad():
        fused_features = fusion_layer(modality_features)
    
    print(f"  Input modalities: {list(modality_dims.keys())}")
    print(f"  Input shapes: {[f.shape for f in modality_features.values()]}")
    print(f"  Output: {fused_features.shape}")
    print(f"  ✓ Hypergraph Fusion working correctly")
    
    return fused_features


def test_multi_objective_loss():
    """Test Multi-Objective Loss component."""
    print("\nTesting Multi-Objective Loss...")
    
    config = Config("configs/config.yaml")
    config.set("data.num_moa_classes", 20)
    
    # Create loss function
    loss_fn = MultiObjectiveLoss(config)
    
    # Create sample data
    batch_size = 4
    num_classes = 20
    embedding_dim = 256
    
    logits = torch.randn(batch_size, num_classes)
    embeddings = torch.randn(batch_size, embedding_dim)
    targets = create_sample_targets(batch_size, num_classes)
    embeddings_aug = torch.randn(batch_size, embedding_dim)  # Augmented embeddings
    
    # Forward pass
    total_loss, loss_components = loss_fn(
        logits=logits,
        embeddings=embeddings,
        targets=targets,
        embeddings_aug=embeddings_aug,
        return_components=True
    )
    
    print(f"  Input: logits {logits.shape}, embeddings {embeddings.shape}, targets {targets.shape}")
    print(f"  Total loss: {total_loss.item():.4f}")
    print(f"  Loss components:")
    for component, value in loss_components.items():
        print(f"    {component}: {value.item():.4f}")
    print(f"  ✓ Multi-Objective Loss working correctly")
    
    return total_loss


def test_complete_model():
    """Test the complete multi-modal model."""
    print("\nTesting Complete Multi-Modal Model...")
    
    config = Config("configs/config.yaml")
    config.set("data.num_moa_classes", 20)
    
    # Enable all modalities
    config.set("scope.modalities.chemistry", True)
    config.set("scope.modalities.targets", True)
    config.set("scope.modalities.pathways", True)
    config.set("scope.modalities.perturbation", True)
    config.set("scope.modalities.structures", True)
    
    # Create model
    model = MultiModalMoAPredictor(config)
    
    # Create sample batch data
    batch_size = 4
    batch_data = {
        "molecular_graphs": create_sample_molecular_graphs(batch_size),
        **create_sample_biological_features(batch_size)
    }
    
    targets = create_sample_targets(batch_size, 20)
    
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Enabled modalities: {list(model.modality_encoders.keys())}")
    
    # Test forward pass
    with torch.no_grad():
        # Basic prediction
        logits = model(batch_data, training=False)
        print(f"  Prediction output: {logits.shape}")
        
        # Get embeddings and attention
        result = model(batch_data, return_embeddings=True, return_attention=True, training=False)
        print(f"  Embeddings: {result['embeddings'].shape}")
        print(f"  Attention weights: {list(result['attention_weights'].keys())}")
        
        # Get modality importance
        importance = model.get_modality_importance(batch_data)
        print(f"  Modality importance: {importance}")
    
    # Test loss computation
    model.train()
    loss = model.compute_loss(batch_data, targets)
    print(f"  Training loss: {loss.item():.4f}")
    
    print(f"  ✓ Complete Multi-Modal Model working correctly")
    
    return model


def test_model_interpretability():
    """Test model interpretability features."""
    print("\nTesting Model Interpretability...")
    
    config = Config("configs/config.yaml")
    config.set("data.num_moa_classes", 20)
    config.set("scope.modalities.chemistry", True)
    config.set("scope.modalities.targets", True)
    config.set("scope.modalities.pathways", True)
    config.set("scope.modalities.perturbation", True)
    
    model = MultiModalMoAPredictor(config)
    
    # Create sample data
    batch_size = 4
    batch_data = {
        "molecular_graphs": create_sample_molecular_graphs(batch_size),
        **create_sample_biological_features(batch_size)
    }
    
    # Get detailed embeddings
    embeddings_dict = model.get_embeddings(batch_data)
    
    print(f"  Final embeddings: {embeddings_dict['final_embeddings'].shape}")
    print(f"  Modality features:")
    for modality, features in embeddings_dict['modality_features'].items():
        print(f"    {modality}: {features.shape}")
    
    # Get learned prototypes
    prototypes = model.loss_fn.get_prototypes()
    print(f"  Learned prototypes: {prototypes.shape}")
    
    # Test attention visualization
    if embeddings_dict['attention_weights']:
        print(f"  Attention weights available for: {list(embeddings_dict['attention_weights'].keys())}")
    
    print(f"  ✓ Model interpretability features working correctly")


def main():
    """Main demonstration function."""
    print("MoA Prediction Framework - Model Architecture Demo")
    print("=" * 60)
    
    try:
        # Test individual components
        test_graph_transformer()
        test_pathway_transformer()
        test_hypergraph_fusion()
        test_multi_objective_loss()
        
        # Test complete model
        model = test_complete_model()
        test_model_interpretability()
        
        print("\n" + "=" * 60)
        print("✓ All model components working correctly!")
        
        # Model summary
        print(f"\nModel Summary:")
        print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"  Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        print(f"  Model size: ~{sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024:.1f} MB")
        
        print(f"\nArchitecture Highlights:")
        print(f"  ✓ Graph Transformer with counterfactual-aware pooling")
        print(f"  ✓ Pathway Transformer with biological hierarchy encoding")
        print(f"  ✓ Hypergraph Neural Networks for multi-modal fusion")
        print(f"  ✓ Multi-objective loss (classification + prototype + invariance + contrastive)")
        print(f"  ✓ Modality dropout for robustness")
        print(f"  ✓ Attention mechanisms for interpretability")
        
        print(f"\nNext Steps:")
        print(f"  1. Implement training pipeline (Phase 4)")
        print(f"  2. Add evaluation metrics and baselines")
        print(f"  3. Build interpretation and visualization tools")
        
    except Exception as e:
        print(f"\n✗ Model architecture demo failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
