"""
Integration tests for Phase 4: Training & Evaluation Pipeline

This module tests the complete training and evaluation pipeline,
including trainer, evaluator, baselines, and statistical testing.
"""

import pytest
import torch
import numpy as np
import tempfile
import shutil
from pathlib import Path
from torch.utils.data import DataLoader
from torch_geometric.data import Data, Batch

from moa.utils.config import Config
from moa.models.multimodal_model import MultiModalMoAPredictor
from moa.training.trainer import MoATrainer
from moa.training.optimization import OptimizerFactory, SchedulerFactory
from moa.training.curriculum import CurriculumLearning
from moa.training.monitoring import TrainingMonitor
from moa.evaluation.evaluator import MoAEvaluator
from moa.evaluation.metrics import MoAMetrics
from moa.evaluation.baselines import BaselineModels
from moa.evaluation.statistical_tests import StatisticalTests


class TestDataset(torch.utils.data.Dataset):
    """Simple test dataset."""
    
    def __init__(self, num_samples=100, num_classes=5):
        self.num_samples = num_samples
        self.num_classes = num_classes
        
        # Create molecular graphs
        self.molecular_graphs = []
        for i in range(num_samples):
            num_nodes = np.random.randint(10, 20)
            num_edges = np.random.randint(num_nodes, num_nodes * 2)
            
            x = torch.randn(num_nodes, 64)
            edge_index = torch.randint(0, num_nodes, (2, num_edges))
            edge_attr = torch.randn(num_edges, 16)
            
            graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
            self.molecular_graphs.append(graph)
        
        # Create biological features
        self.biological_features = {
            'mechtoken_features': torch.randn(num_samples, 128),
            'gene_signature_features': torch.randn(num_samples, 978),
            'pathway_score_features': torch.randn(num_samples, 50)
        }
        
        # Create sparse multi-label targets
        self.targets = torch.zeros(num_samples, num_classes)
        for i in range(num_samples):
            num_positive = np.random.randint(1, 3)
            positive_indices = np.random.choice(num_classes, num_positive, replace=False)
            self.targets[i, positive_indices] = 1.0
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        batch_data = {
            'molecular_graphs': self.molecular_graphs[idx],
            **{k: v[idx] for k, v in self.biological_features.items()}
        }
        return batch_data, self.targets[idx]


def collate_fn(batch):
    """Custom collate function for test data."""
    batch_data_list, targets_list = zip(*batch)
    
    # Collate targets
    targets = torch.stack(targets_list, dim=0)
    
    # Collate molecular graphs
    graphs = [data['molecular_graphs'] for data in batch_data_list]
    batched_graphs = Batch.from_data_list(graphs)
    
    # Collate biological features
    collated_batch_data = {'molecular_graphs': batched_graphs}
    for feature_name in ['mechtoken_features', 'gene_signature_features', 'pathway_score_features']:
        features = [data[feature_name] for data in batch_data_list]
        collated_batch_data[feature_name] = torch.stack(features, dim=0)
    
    return collated_batch_data, targets


@pytest.fixture
def test_config():
    """Create test configuration."""
    config_dict = {
        'data': {
            'num_moa_classes': 5,
            'batch_size': 8
        },
        'model': {
            'chemical_features': {
                'node_dim': 64,
                'edge_dim': 16,
                'hidden_dim': 128,
                'num_layers': 2
            },
            'biological_features': {
                'mechtoken_dim': 128,
                'gene_signature_dim': 978,
                'pathway_score_dim': 50,
                'hidden_dim': 128
            },
            'fusion': {
                'use_hypergraph': True,
                'hidden_dim': 256,
                'num_layers': 2
            }
        },
        'training': {
            'num_epochs': 3,
            'batch_size': 8,
            'eval_batch_size': 16,
            'optimizer': {
                'name': 'adam',
                'learning_rate': 1e-3
            },
            'scheduler': {
                'name': 'cosine',
                'T_max': 10
            },
            'curriculum_learning': {
                'enable': True,
                'strategy': 'linear'
            },
            'early_stopping': {
                'patience': 5,
                'min_delta': 1e-4
            }
        },
        'evaluation': {
            'metrics': ['auroc', 'f1', 'precision', 'recall'],
            'save_predictions': True,
            'generate_plots': False
        }
    }
    
    return Config(config_dict)


@pytest.fixture
def test_data_loaders():
    """Create test data loaders."""
    train_dataset = TestDataset(num_samples=80, num_classes=5)
    val_dataset = TestDataset(num_samples=20, num_classes=5)
    test_dataset = TestDataset(num_samples=20, num_classes=5)
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)
    
    return train_loader, val_loader, test_loader


@pytest.fixture
def test_model(test_config):
    """Create test model."""
    return MultiModalMoAPredictor(test_config)


class TestTrainingPipeline:
    """Test training pipeline components."""
    
    def test_trainer_initialization(self, test_model, test_config, test_data_loaders):
        """Test trainer initialization."""
        train_loader, val_loader, test_loader = test_data_loaders
        
        trainer = MoATrainer(
            model=test_model,
            config=test_config,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader
        )
        
        assert trainer.model is not None
        assert trainer.optimizer is not None
        assert trainer.criterion is not None
        assert trainer.device is not None
        assert trainer.use_curriculum == test_config.get('training.curriculum_learning.enable')
    
    def test_training_step(self, test_model, test_config, test_data_loaders):
        """Test single training step."""
        train_loader, val_loader, test_loader = test_data_loaders
        
        trainer = MoATrainer(
            model=test_model,
            config=test_config,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader
        )
        
        # Get a batch
        batch_data, targets = next(iter(train_loader))
        
        # Perform training step
        loss, metrics = trainer.training_step(batch_data, targets)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.item() > 0
        assert isinstance(metrics, dict)
        assert 'train_loss' in metrics
    
    def test_validation_step(self, test_model, test_config, test_data_loaders):
        """Test validation step."""
        train_loader, val_loader, test_loader = test_data_loaders
        
        trainer = MoATrainer(
            model=test_model,
            config=test_config,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader
        )
        
        # Perform validation
        val_metrics = trainer.validate_epoch()
        
        assert isinstance(val_metrics, dict)
        assert 'val_loss' in val_metrics
        assert 'val_auroc_macro' in val_metrics
    
    def test_full_training(self, test_model, test_config, test_data_loaders):
        """Test complete training pipeline."""
        train_loader, val_loader, test_loader = test_data_loaders
        
        trainer = MoATrainer(
            model=test_model,
            config=test_config,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader
        )
        
        # Train model
        training_summary = trainer.train()
        
        assert isinstance(training_summary, dict)
        assert 'best_val_score' in training_summary
        assert 'total_epochs' in training_summary
        assert 'total_steps' in training_summary
        assert training_summary['total_epochs'] <= test_config.get('training.num_epochs')


class TestEvaluationFramework:
    """Test evaluation framework components."""
    
    def test_metrics_calculator(self):
        """Test metrics calculator."""
        metrics_calc = MoAMetrics()
        
        # Create test data
        y_true = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 0]])
        y_pred = np.array([[0.8, 0.2, 0.9], [0.1, 0.7, 0.3], [0.6, 0.8, 0.1]])
        y_prob = y_pred.copy()
        
        # Compute metrics
        metrics = metrics_calc.compute_moa_specific_metrics(y_true, y_pred, y_prob)
        
        assert isinstance(metrics, dict)
        assert 'auroc_macro' in metrics
        assert 'f1_macro' in metrics
        assert 'precision_macro' in metrics
        assert 'recall_macro' in metrics
        assert 'subset_accuracy' in metrics
        assert 'hamming_loss' in metrics
    
    def test_evaluator_initialization(self, test_config):
        """Test evaluator initialization."""
        moa_classes = [f"MoA_{i}" for i in range(5)]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            evaluator = MoAEvaluator(
                config=test_config,
                moa_classes=moa_classes,
                output_dir=temp_dir
            )
            
            assert evaluator.config is not None
            assert evaluator.moa_classes == moa_classes
            assert evaluator.output_dir == temp_dir
            assert evaluator.metrics_calculator is not None
    
    def test_model_evaluation(self, test_model, test_config, test_data_loaders):
        """Test model evaluation."""
        _, _, test_loader = test_data_loaders
        moa_classes = [f"MoA_{i}" for i in range(5)]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            evaluator = MoAEvaluator(
                config=test_config,
                moa_classes=moa_classes,
                output_dir=temp_dir
            )
            
            # Evaluate model
            result = evaluator.evaluate_model(
                model=test_model,
                data_loader=test_loader,
                model_name="TestModel"
            )
            
            assert result.model_name == "TestModel"
            assert isinstance(result.metrics, dict)
            assert 'auroc_macro' in result.metrics
            assert result.true_labels is not None
            assert result.predictions is not None
            assert result.probabilities is not None


class TestBaselineModels:
    """Test baseline model components."""
    
    def test_baseline_factory(self, test_config):
        """Test baseline models factory."""
        baseline_factory = BaselineModels(test_config)
        
        available_baselines = baseline_factory.get_available_baselines()
        assert isinstance(available_baselines, list)
        assert len(available_baselines) > 0
    
    def test_baseline_creation(self, test_config):
        """Test baseline model creation."""
        baseline_factory = BaselineModels(test_config)
        available_baselines = baseline_factory.get_available_baselines()
        
        if available_baselines:
            baseline_name = available_baselines[0]
            baseline_model = baseline_factory.create_baseline(baseline_name)
            
            assert baseline_model is not None
            assert baseline_model.name == baseline_name
            assert not baseline_model.is_fitted


class TestStatisticalTesting:
    """Test statistical testing components."""
    
    def test_statistical_tests_initialization(self):
        """Test statistical tests initialization."""
        stat_tests = StatisticalTests(alpha=0.05)
        
        assert stat_tests.alpha == 0.05
    
    def test_model_comparison(self):
        """Test model comparison."""
        stat_tests = StatisticalTests(alpha=0.05)
        
        # Create test data
        n_samples, n_classes = 50, 3
        y_true = np.random.randint(0, 2, (n_samples, n_classes))
        y_pred1 = np.random.rand(n_samples, n_classes)
        y_pred2 = np.random.rand(n_samples, n_classes)
        
        # Perform Wilcoxon test
        result = stat_tests.compare_models(
            y_pred1, y_pred2, y_true,
            test_type="wilcoxon", metric="auroc"
        )
        
        assert isinstance(result, dict)
        assert 'p_value' in result
        assert 'significant' in result
        assert 'effect_size' in result
        assert isinstance(result['significant'], bool)
    
    def test_bootstrap_confidence_interval(self):
        """Test bootstrap confidence interval."""
        stat_tests = StatisticalTests(alpha=0.05)
        
        # Create test data
        y_true = np.array([[1, 0], [0, 1], [1, 1], [0, 0]])
        y_pred = np.array([[0.8, 0.2], [0.1, 0.9], [0.7, 0.8], [0.3, 0.1]])
        
        def dummy_metric(y_true, y_pred):
            return np.mean(y_pred)
        
        # Compute confidence interval
        metric_val, lower_ci, upper_ci = stat_tests.bootstrap_confidence_interval(
            y_true, y_pred, dummy_metric, n_bootstrap=10
        )
        
        assert isinstance(metric_val, float)
        assert isinstance(lower_ci, float)
        assert isinstance(upper_ci, float)
        assert lower_ci <= metric_val <= upper_ci
    
    def test_multiple_comparisons_correction(self):
        """Test multiple comparisons correction."""
        stat_tests = StatisticalTests(alpha=0.05)
        
        p_values = [0.01, 0.03, 0.08, 0.12, 0.45]
        
        # Bonferroni correction
        corrected_p, rejected = stat_tests.multiple_comparisons_correction(
            p_values, method="bonferroni"
        )
        
        assert len(corrected_p) == len(p_values)
        assert len(rejected) == len(p_values)
        assert all(cp >= p for cp, p in zip(corrected_p, p_values))
        assert all(isinstance(r, bool) for r in rejected)


class TestIntegration:
    """Integration tests for complete pipeline."""
    
    def test_end_to_end_pipeline(self, test_config):
        """Test complete end-to-end pipeline."""
        # Create data
        train_dataset = TestDataset(num_samples=40, num_classes=5)
        val_dataset = TestDataset(num_samples=10, num_classes=5)
        test_dataset = TestDataset(num_samples=10, num_classes=5)
        
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)
        
        # Create model
        model = MultiModalMoAPredictor(test_config)
        
        # Train model
        trainer = MoATrainer(
            model=model,
            config=test_config,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader
        )
        
        training_summary = trainer.train()
        assert training_summary['total_epochs'] > 0
        
        # Evaluate model
        moa_classes = [f"MoA_{i}" for i in range(5)]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            evaluator = MoAEvaluator(
                config=test_config,
                moa_classes=moa_classes,
                output_dir=temp_dir
            )
            
            result = evaluator.evaluate_model(
                model=model,
                data_loader=test_loader,
                model_name="EndToEndTest"
            )
            
            assert result.model_name == "EndToEndTest"
            assert 'auroc_macro' in result.metrics
            assert result.metrics['auroc_macro'] >= 0.0
            assert result.metrics['auroc_macro'] <= 1.0
    
    def test_reproducibility(self, test_config):
        """Test training reproducibility."""
        # Set seeds
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Create identical datasets
        train_dataset1 = TestDataset(num_samples=40, num_classes=5)
        train_dataset2 = TestDataset(num_samples=40, num_classes=5)
        
        # Reset seeds to ensure identical data
        torch.manual_seed(42)
        np.random.seed(42)
        train_dataset2 = TestDataset(num_samples=40, num_classes=5)
        
        # Create data loaders
        train_loader1 = DataLoader(train_dataset1, batch_size=8, shuffle=False, collate_fn=collate_fn)
        train_loader2 = DataLoader(train_dataset2, batch_size=8, shuffle=False, collate_fn=collate_fn)
        
        # Create identical models
        torch.manual_seed(42)
        model1 = MultiModalMoAPredictor(test_config)
        
        torch.manual_seed(42)
        model2 = MultiModalMoAPredictor(test_config)
        
        # Check initial parameters are identical
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            assert torch.allclose(p1, p2, atol=1e-6)
        
        print("Phase 4 integration tests completed successfully!")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
