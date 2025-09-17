"""Tests for model architecture components."""

import pytest
import torch
import torch.nn as nn
import numpy as np
from torch_geometric.data import Data, Batch

from moa.utils.config import Config
from moa.models.graph_transformer import GraphTransformer, GraphTransformerLayer
from moa.models.pathway_transformer import PathwayTransformer, BiologicalHierarchyEncoder
from moa.models.hypergraph_layers import HypergraphConv, HypergraphFusion
from moa.models.losses import MultiObjectiveLoss, ClassificationLoss, PrototypeLoss
from moa.models.multimodal_model import MultiModalMoAPredictor


@pytest.fixture
def sample_config():
    """Create a sample configuration for testing."""
    config_dict = {
        "data": {
            "num_moa_classes": 10
        },
        "models": {
            "embedding_dim": 128,
            "dropout": 0.1,
            "use_hypergraph_fusion": True,
            "graph_transformer": {
                "num_layers": 2,
                "hidden_dim": 128,
                "num_heads": 4,
                "dropout": 0.1,
                "pooling": "attention",
                "use_counterfactual": True
            },
            "pathway_transformer": {
                "num_layers": 2,
                "hidden_dim": 128,
                "num_heads": 4,
                "dropout": 0.1,
                "use_hierarchy": True,
                "use_pathway_bias": True
            },
            "hypergraph": {
                "num_layers": 2,
                "num_heads": 4,
                "dropout": 0.1
            }
        },
        "features": {
            "chemistry": {
                "graph_features": {
                    "node_dim": 32,
                    "edge_dim": 8
                }
            },
            "mechanism_tokens": {
                "embedding_dim": 64
            },
            "perturbation": {
                "gene_signature_dim": 100,
                "pathway_score_dim": 20
            }
        },
        "training": {
            "loss_weights": {
                "classification": 1.0,
                "prototype": 0.5,
                "invariance": 0.3,
                "contrastive": 0.2
            },
            "use_focal_loss": True,
            "focal_alpha": 0.25,
            "focal_gamma": 2.0,
            "prototype_temperature": 0.1,
            "prototype_margin": 0.5,
            "invariance_type": "augmentation",
            "invariance_temperature": 0.1,
            "lambda_invariance": 1.0,
            "contrastive_temperature": 0.1,
            "contrastive_margin": 0.5,
            "negative_sampling": "hard",
            "modality_dropout": 0.1
        },
        "scope": {
            "modalities": {
                "chemistry": True,
                "targets": True,
                "pathways": True,
                "perturbation": True,
                "structures": False
            }
        }
    }
    
    # Create temporary config
    import tempfile
    import yaml
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config_dict, f)
        config_path = f.name
    
    config = Config(config_path)
    yield config
    
    # Cleanup
    import os
    os.unlink(config_path)


@pytest.fixture
def sample_molecular_graphs():
    """Create sample molecular graphs."""
    graphs = []
    for i in range(3):
        num_nodes = 10 + i * 5
        num_edges = num_nodes + i * 3
        
        x = torch.randn(num_nodes, 32)  # Node features
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        edge_attr = torch.randn(num_edges, 8)  # Edge features
        
        graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        graphs.append(graph)
    
    return Batch.from_data_list(graphs)


@pytest.fixture
def sample_biological_features():
    """Create sample biological features."""
    batch_size = 3
    return {
        "mechtoken_features": torch.randn(batch_size, 64),
        "gene_signature_features": torch.randn(batch_size, 100),
        "pathway_score_features": torch.randn(batch_size, 20)
    }


@pytest.fixture
def sample_targets():
    """Create sample multi-label targets."""
    batch_size = 3
    num_classes = 10
    targets = torch.zeros(batch_size, num_classes)
    
    # Create sparse multi-label targets
    for i in range(batch_size):
        num_positive = np.random.randint(1, 4)
        positive_indices = np.random.choice(num_classes, num_positive, replace=False)
        targets[i, positive_indices] = 1.0
    
    return targets


class TestGraphTransformer:
    """Test Graph Transformer components."""
    
    def test_graph_transformer_layer(self, sample_config):
        """Test GraphTransformerLayer."""
        layer = GraphTransformerLayer(
            in_channels=32,
            out_channels=64,
            heads=4,
            dropout=0.1,
            edge_dim=8
        )
        
        # Create sample data
        x = torch.randn(20, 32)
        edge_index = torch.randint(0, 20, (2, 30))
        edge_attr = torch.randn(30, 8)
        batch = torch.zeros(20, dtype=torch.long)
        
        # Forward pass
        out = layer(x, edge_index, edge_attr, batch)
        
        assert out.shape == (20, 64)
        assert not torch.isnan(out).any()
    
    def test_graph_transformer(self, sample_config, sample_molecular_graphs):
        """Test complete GraphTransformer."""
        transformer = GraphTransformer(sample_config)
        
        # Forward pass
        embeddings = transformer(
            sample_molecular_graphs.x,
            sample_molecular_graphs.edge_index,
            sample_molecular_graphs.edge_attr,
            sample_molecular_graphs.batch
        )
        
        batch_size = sample_molecular_graphs.batch.max().item() + 1
        assert embeddings.shape[0] == batch_size
        assert embeddings.shape[1] == transformer.output_dim
        assert not torch.isnan(embeddings).any()
    
    def test_graph_transformer_with_attention(self, sample_config, sample_molecular_graphs):
        """Test GraphTransformer with attention weights."""
        transformer = GraphTransformer(sample_config)
        
        # Forward pass with attention
        embeddings, attention_weights = transformer(
            sample_molecular_graphs.x,
            sample_molecular_graphs.edge_index,
            sample_molecular_graphs.edge_attr,
            sample_molecular_graphs.batch,
            return_attention=True
        )
        
        assert len(attention_weights) == transformer.num_layers
        assert embeddings.shape[0] == sample_molecular_graphs.batch.max().item() + 1


class TestPathwayTransformer:
    """Test Pathway Transformer components."""
    
    def test_biological_hierarchy_encoder(self):
        """Test BiologicalHierarchyEncoder."""
        encoder = BiologicalHierarchyEncoder(
            input_dim=64,
            hidden_dim=128,
            num_levels=3,
            dropout=0.1
        )
        
        # Create sample data
        x = torch.randn(2, 10, 64)  # [batch, seq, features]
        
        # Forward pass
        encoded, level_outputs = encoder(x)
        
        assert encoded.shape == (2, 10, 128)
        assert len(level_outputs) == 3
        assert not torch.isnan(encoded).any()
    
    def test_pathway_transformer(self, sample_config, sample_biological_features):
        """Test complete PathwayTransformer."""
        transformer = PathwayTransformer(sample_config)
        
        # Forward pass
        embeddings = transformer(
            sample_biological_features["mechtoken_features"],
            sample_biological_features["gene_signature_features"],
            sample_biological_features["pathway_score_features"]
        )
        
        batch_size = sample_biological_features["mechtoken_features"].shape[0]
        assert embeddings.shape == (batch_size, transformer.output_dim)
        assert not torch.isnan(embeddings).any()
    
    def test_pathway_transformer_with_attention(self, sample_config, sample_biological_features):
        """Test PathwayTransformer with attention weights."""
        transformer = PathwayTransformer(sample_config)
        
        # Forward pass with attention
        embeddings, attention_weights = transformer(
            sample_biological_features["mechtoken_features"],
            sample_biological_features["gene_signature_features"],
            sample_biological_features["pathway_score_features"],
            return_attention=True
        )
        
        assert len(attention_weights) == transformer.num_layers
        batch_size = sample_biological_features["mechtoken_features"].shape[0]
        assert embeddings.shape == (batch_size, transformer.output_dim)


class TestHypergraphLayers:
    """Test Hypergraph Neural Network components."""
    
    def test_hypergraph_conv(self):
        """Test HypergraphConv layer."""
        layer = HypergraphConv(
            in_channels=64,
            out_channels=128,
            use_attention=True,
            dropout=0.1
        )
        
        # Create sample data
        x = torch.randn(10, 64)  # Node features
        hyperedge_index = torch.randint(0, 10, (2, 15))  # Hyperedge connections
        hyperedge_attr = torch.randn(15, 64)  # Hyperedge attributes
        
        # Forward pass
        out = layer(x, hyperedge_index, hyperedge_attr)
        
        assert out.shape == (10, 128)
        assert not torch.isnan(out).any()
    
    def test_hypergraph_fusion(self):
        """Test HypergraphFusion layer."""
        modality_dims = {
            "chemistry": 128,
            "biology": 128,
            "structure": 128
        }
        
        fusion = HypergraphFusion(
            modality_dims=modality_dims,
            hidden_dim=256,
            num_hypergraph_layers=2,
            num_attention_heads=4
        )
        
        # Create sample modality features
        batch_size = 3
        modality_features = {
            modality: torch.randn(batch_size, dim)
            for modality, dim in modality_dims.items()
        }
        
        # Forward pass
        fused_features = fusion(modality_features)
        
        assert fused_features.shape == (batch_size, 256)
        assert not torch.isnan(fused_features).any()


class TestLossFunctions:
    """Test loss function components."""
    
    def test_classification_loss(self, sample_targets):
        """Test ClassificationLoss."""
        loss_fn = ClassificationLoss(
            num_classes=10,
            use_focal=True,
            focal_alpha=0.25,
            focal_gamma=2.0
        )
        
        batch_size = sample_targets.shape[0]
        logits = torch.randn(batch_size, 10)
        
        loss = loss_fn(logits, sample_targets)
        
        assert loss.item() > 0
        assert not torch.isnan(loss)
    
    def test_prototype_loss(self, sample_targets):
        """Test PrototypeLoss."""
        loss_fn = PrototypeLoss(
            num_classes=10,
            embedding_dim=128,
            temperature=0.1,
            margin=0.5
        )
        
        batch_size = sample_targets.shape[0]
        embeddings = torch.randn(batch_size, 128)
        
        loss = loss_fn(embeddings, sample_targets)
        
        assert loss.item() > 0
        assert not torch.isnan(loss)
        assert loss_fn.prototypes.shape == (10, 128)
    
    def test_multi_objective_loss(self, sample_config, sample_targets):
        """Test MultiObjectiveLoss."""
        loss_fn = MultiObjectiveLoss(sample_config)
        
        batch_size = sample_targets.shape[0]
        logits = torch.randn(batch_size, 10)
        embeddings = torch.randn(batch_size, 128)
        embeddings_aug = torch.randn(batch_size, 128)
        
        # Test with components
        total_loss, components = loss_fn(
            logits=logits,
            embeddings=embeddings,
            targets=sample_targets,
            embeddings_aug=embeddings_aug,
            return_components=True
        )
        
        assert total_loss.item() > 0
        assert not torch.isnan(total_loss)
        assert "classification" in components
        assert "prototype" in components
        assert "invariance" in components
        assert "contrastive" in components


class TestMultiModalModel:
    """Test complete multi-modal model."""
    
    def test_model_initialization(self, sample_config):
        """Test model initialization."""
        model = MultiModalMoAPredictor(sample_config)
        
        assert "chemistry" in model.modality_encoders
        assert "biology" in model.modality_encoders
        assert hasattr(model, "fusion_layer")
        assert hasattr(model, "prediction_head")
        assert hasattr(model, "loss_fn")
    
    def test_model_forward(self, sample_config, sample_molecular_graphs, sample_biological_features):
        """Test model forward pass."""
        model = MultiModalMoAPredictor(sample_config)
        
        batch_data = {
            "molecular_graphs": sample_molecular_graphs,
            **sample_biological_features
        }
        
        # Basic forward pass
        logits = model(batch_data, training=False)
        
        batch_size = sample_molecular_graphs.batch.max().item() + 1
        assert logits.shape == (batch_size, 10)
        assert not torch.isnan(logits).any()
    
    def test_model_with_embeddings(self, sample_config, sample_molecular_graphs, sample_biological_features):
        """Test model forward pass with embeddings."""
        model = MultiModalMoAPredictor(sample_config)
        
        batch_data = {
            "molecular_graphs": sample_molecular_graphs,
            **sample_biological_features
        }
        
        # Forward pass with embeddings
        logits, embeddings = model(batch_data, return_embeddings=True, training=False)
        
        batch_size = sample_molecular_graphs.batch.max().item() + 1
        assert logits.shape == (batch_size, 10)
        assert embeddings.shape == (batch_size, 128)
        assert not torch.isnan(logits).any()
        assert not torch.isnan(embeddings).any()
    
    def test_model_loss_computation(self, sample_config, sample_molecular_graphs, 
                                   sample_biological_features, sample_targets):
        """Test model loss computation."""
        model = MultiModalMoAPredictor(sample_config)
        
        batch_data = {
            "molecular_graphs": sample_molecular_graphs,
            **sample_biological_features
        }
        
        # Compute loss
        loss = model.compute_loss(batch_data, sample_targets)
        
        assert loss.item() > 0
        assert not torch.isnan(loss)
    
    def test_model_prediction(self, sample_config, sample_molecular_graphs, sample_biological_features):
        """Test model prediction."""
        model = MultiModalMoAPredictor(sample_config)
        
        batch_data = {
            "molecular_graphs": sample_molecular_graphs,
            **sample_biological_features
        }
        
        # Make predictions
        probabilities = model.predict(batch_data, return_probabilities=True)
        logits = model.predict(batch_data, return_probabilities=False)
        
        batch_size = sample_molecular_graphs.batch.max().item() + 1
        assert probabilities.shape == (batch_size, 10)
        assert logits.shape == (batch_size, 10)
        assert torch.all(probabilities >= 0) and torch.all(probabilities <= 1)
    
    def test_model_interpretability(self, sample_config, sample_molecular_graphs, sample_biological_features):
        """Test model interpretability features."""
        model = MultiModalMoAPredictor(sample_config)
        
        batch_data = {
            "molecular_graphs": sample_molecular_graphs,
            **sample_biological_features
        }
        
        # Get embeddings
        embeddings_dict = model.get_embeddings(batch_data)
        
        assert "final_embeddings" in embeddings_dict
        assert "modality_features" in embeddings_dict
        
        # Get modality importance
        importance = model.get_modality_importance(batch_data)
        
        assert isinstance(importance, dict)
        assert len(importance) > 0


def test_integration_model_architecture(sample_config, sample_molecular_graphs, 
                                       sample_biological_features, sample_targets):
    """Test the complete model architecture integration."""
    model = MultiModalMoAPredictor(sample_config)
    
    batch_data = {
        "molecular_graphs": sample_molecular_graphs,
        **sample_biological_features
    }
    
    # Test training mode
    model.train()
    loss = model.compute_loss(batch_data, sample_targets)
    assert loss.item() > 0
    
    # Test evaluation mode
    model.eval()
    with torch.no_grad():
        logits = model(batch_data, training=False)
        probabilities = model.predict(batch_data)
        embeddings_dict = model.get_embeddings(batch_data)
    
    batch_size = sample_molecular_graphs.batch.max().item() + 1
    assert logits.shape == (batch_size, 10)
    assert probabilities.shape == (batch_size, 10)
    assert embeddings_dict["final_embeddings"].shape == (batch_size, 128)


if __name__ == "__main__":
    pytest.main([__file__])
