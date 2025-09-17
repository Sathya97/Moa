"""
Tests for Phase 5: Interpretation & Applications

This module tests the interpretation and application capabilities
including model explainability, drug repurposing, knowledge discovery,
and therapeutic insights.
"""

import pytest
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import os

from moa.utils.config import Config
from moa.models.multimodal_model import MultiModalMoAModel
from moa.interpretation.explainer import MoAExplainer
from moa.interpretation.uncertainty import UncertaintyEstimator
from moa.applications.drug_repurposing import DrugRepurposingPipeline
from moa.applications.knowledge_discovery import KnowledgeDiscovery
from moa.applications.therapeutic_insights import TherapeuticInsights


@pytest.fixture
def config():
    """Create test configuration."""
    config_data = {
        'model': {
            'num_classes': 20,
            'chemical': {'node_dim': 64, 'edge_dim': 16, 'hidden_dim': 128},
            'biological': {'gene_dim': 978, 'pathway_dim': 50, 'hidden_dim': 128},
            'protein': {'pocket_dim': 256, 'hidden_dim': 128},
            'fusion': {'hidden_dim': 256, 'num_heads': 8}
        },
        'applications': {
            'repurposing': {
                'similarity_threshold': 0.7,
                'top_k_candidates': 10,
                'min_confidence': 0.5
            },
            'discovery': {
                'significance_threshold': 0.05,
                'min_support': 3,
                'clustering_eps': 0.3,
                'min_cluster_size': 2
            },
            'therapeutic': {
                'efficacy_threshold': 0.7,
                'safety_threshold': 0.8,
                'synergy_threshold': 0.6
            }
        }
    }
    return Config(config_data)


@pytest.fixture
def model(config):
    """Create test model."""
    model = MultiModalMoAModel(config)
    model.eval()
    return model


@pytest.fixture
def moa_classes():
    """Create test MoA classes."""
    return [f'MoA_{i}' for i in range(20)]


@pytest.fixture
def demo_compound_data():
    """Create demo compound data for testing."""
    compound_data = {
        'chemical': {
            'node_features': torch.randn(20, 64),
            'edge_index': torch.randint(0, 20, (2, 40)),
            'edge_features': torch.randn(40, 16),
            'batch': torch.zeros(20, dtype=torch.long),
            'counterfactual_weights': torch.rand(20)
        },
        'biological': {
            'gene_expression': torch.randn(978),
            'pathway_scores': torch.randn(50),
            'mechanism_tokens': torch.randn(128)
        },
        'protein': {
            'pocket_features': torch.randn(256)
        }
    }
    return compound_data


@pytest.fixture
def demo_compounds_list():
    """Create list of demo compounds."""
    compounds = []
    for i in range(10):
        compound_data = {
            'chemical': {
                'node_features': torch.randn(15 + i, 64),
                'edge_index': torch.randint(0, 15 + i, (2, 30 + i * 2)),
                'edge_features': torch.randn(30 + i * 2, 16),
                'batch': torch.zeros(15 + i, dtype=torch.long),
                'counterfactual_weights': torch.rand(15 + i)
            },
            'biological': {
                'gene_expression': torch.randn(978),
                'pathway_scores': torch.randn(50),
                'mechanism_tokens': torch.randn(128)
            },
            'protein': {
                'pocket_features': torch.randn(256)
            }
        }
        compounds.append(compound_data)
    return compounds


@pytest.fixture
def demo_metadata():
    """Create demo compound metadata."""
    metadata = []
    for i in range(10):
        meta = {
            'compound_id': f'TEST_{i:03d}',
            'compound_name': f'Test Compound {i+1}',
            'smiles': f'CC(C)C{i}',
            'molecular_weight': 200 + i * 10,
            'drug_class': ['kinase_inhibitor', 'gpcr_agonist', 'ion_channel_blocker'][i % 3],
            'indication': ['cancer', 'diabetes', 'hypertension'][i % 3],
            'development_stage': ['approved', 'clinical', 'preclinical'][i % 3],
            'is_approved_drug': i < 3,
            'safety_profile': {
                'known_toxicities': ['hepatotoxicity'] if i % 4 == 0 else [],
                'contraindications': []
            }
        }
        metadata.append(meta)
    return metadata


class TestMoAExplainer:
    """Test MoA explainer functionality."""
    
    def test_explainer_initialization(self, model, config, moa_classes):
        """Test explainer initialization."""
        explainer = MoAExplainer(model, config, moa_classes)
        
        assert explainer.model == model
        assert explainer.config == config
        assert explainer.moa_classes == moa_classes
        assert explainer.top_k_features == 20
    
    def test_explain_prediction(self, model, config, moa_classes, demo_compound_data):
        """Test single prediction explanation."""
        explainer = MoAExplainer(model, config, moa_classes)
        
        explanation = explainer.explain_prediction(demo_compound_data, top_k_features=5)
        
        # Check explanation structure
        assert 'top_predictions' in explanation
        assert 'prediction_confidence' in explanation
        assert 'model_certainty' in explanation
        assert 'modality_contributions' in explanation
        assert 'feature_importance' in explanation
        
        # Check top predictions
        assert len(explanation['top_predictions']) <= 5
        for pred in explanation['top_predictions']:
            assert 'moa_name' in pred
            assert 'score' in pred
            assert 'confidence' in pred
        
        # Check modality contributions
        modality_contrib = explanation['modality_contributions']
        assert 'chemical' in modality_contrib
        assert 'biological' in modality_contrib
        assert 'protein' in modality_contrib
        
        # Check feature importance
        assert len(explanation['feature_importance']) <= 5
        for feature in explanation['feature_importance']:
            assert 'feature_name' in feature
            assert 'importance' in feature
            assert 'modality' in feature
    
    def test_explain_batch(self, model, config, moa_classes, demo_compounds_list):
        """Test batch explanation."""
        explainer = MoAExplainer(model, config, moa_classes)
        
        explanations = explainer.explain_batch(demo_compounds_list[:3], top_k_features=3)
        
        assert len(explanations) == 3
        for explanation in explanations:
            assert 'top_predictions' in explanation
            assert 'modality_contributions' in explanation
            assert 'feature_importance' in explanation


class TestUncertaintyEstimator:
    """Test uncertainty estimation functionality."""
    
    def test_uncertainty_estimator_initialization(self, model, config):
        """Test uncertainty estimator initialization."""
        estimator = UncertaintyEstimator(model, config)
        
        assert estimator.model == model
        assert estimator.config == config
        assert estimator.n_samples == 100
        assert estimator.dropout_rate == 0.1
    
    def test_estimate_uncertainty(self, model, config, demo_compound_data):
        """Test uncertainty estimation."""
        estimator = UncertaintyEstimator(model, config)
        
        uncertainty_results = estimator.estimate_uncertainty(
            demo_compound_data, n_samples=10
        )
        
        # Check uncertainty results structure
        assert 'epistemic_uncertainty' in uncertainty_results
        assert 'aleatoric_uncertainty' in uncertainty_results
        assert 'total_uncertainty' in uncertainty_results
        assert 'prediction_confidence' in uncertainty_results
        assert 'prediction_samples' in uncertainty_results
        
        # Check uncertainty values are reasonable
        assert 0 <= uncertainty_results['epistemic_uncertainty'] <= 1
        assert 0 <= uncertainty_results['aleatoric_uncertainty'] <= 1
        assert 0 <= uncertainty_results['total_uncertainty'] <= 1
        assert 0 <= uncertainty_results['prediction_confidence'] <= 1
        
        # Check prediction samples
        assert len(uncertainty_results['prediction_samples']) == 10
    
    def test_calibrate_predictions(self, model, config, demo_compounds_list):
        """Test prediction calibration."""
        estimator = UncertaintyEstimator(model, config)
        
        # Create dummy true labels
        true_labels = [torch.randint(0, 2, (20,)).float() for _ in range(5)]
        
        calibration_results = estimator.calibrate_predictions(
            demo_compounds_list[:5], true_labels
        )
        
        assert 'calibrated_model' in calibration_results
        assert 'calibration_curve' in calibration_results
        assert 'ece_score' in calibration_results
        assert 'reliability_diagram' in calibration_results


class TestDrugRepurposingPipeline:
    """Test drug repurposing pipeline functionality."""
    
    def test_repurposing_pipeline_initialization(self, model, config, moa_classes):
        """Test repurposing pipeline initialization."""
        pipeline = DrugRepurposingPipeline(model, config, moa_classes)
        
        assert pipeline.model == model
        assert pipeline.config == config
        assert pipeline.moa_classes == moa_classes
        assert pipeline.similarity_threshold == 0.7
        assert pipeline.top_k_candidates == 10
    
    def test_identify_repurposing_candidates(
        self, model, config, moa_classes, demo_compounds_list, demo_metadata
    ):
        """Test repurposing candidate identification."""
        pipeline = DrugRepurposingPipeline(model, config, moa_classes)
        
        query_compound = demo_compounds_list[0]
        candidate_compounds = demo_compounds_list[1:6]
        candidate_metadata = demo_metadata[1:6]
        
        results = pipeline.identify_repurposing_candidates(
            query_compound_data=query_compound,
            target_disease="Test Disease",
            candidate_compounds=candidate_compounds,
            compound_metadata=candidate_metadata
        )
        
        # Check results structure
        assert 'target_disease' in results
        assert 'query_moa_profile' in results
        assert 'ranked_candidates' in results
        assert 'similarities' in results
        assert 'hypotheses' in results
        assert 'confidence_scores' in results
        assert 'summary_statistics' in results
        
        # Check ranked candidates
        assert len(results['ranked_candidates']) <= 10
        for candidate in results['ranked_candidates']:
            assert 'candidate_index' in candidate
            assert 'ranking_score' in candidate
            assert 'similarity_metrics' in candidate
            assert 'moa_profile' in candidate
        
        # Check hypotheses
        for hypothesis in results['hypotheses']:
            assert 'hypothesis_text' in hypothesis
            assert 'confidence_level' in hypothesis
            assert 'shared_moas' in hypothesis
    
    def test_create_repurposing_network(
        self, model, config, moa_classes, demo_compounds_list, demo_metadata
    ):
        """Test repurposing network creation."""
        pipeline = DrugRepurposingPipeline(model, config, moa_classes)
        
        # Get repurposing results first
        query_compound = demo_compounds_list[0]
        candidate_compounds = demo_compounds_list[1:4]
        candidate_metadata = demo_metadata[1:4]
        
        results = pipeline.identify_repurposing_candidates(
            query_compound_data=query_compound,
            target_disease="Test Disease",
            candidate_compounds=candidate_compounds,
            compound_metadata=candidate_metadata
        )
        
        # Create network
        network = pipeline.create_repurposing_network(results)
        
        assert network.number_of_nodes() > 0
        assert network.number_of_edges() > 0
        
        # Check node types
        node_types = set()
        for node, data in network.nodes(data=True):
            node_types.add(data.get('node_type'))
        
        assert 'query' in node_types
        assert 'candidate' in node_types or 'moa' in node_types


class TestKnowledgeDiscovery:
    """Test knowledge discovery functionality."""
    
    def test_knowledge_discovery_initialization(self, model, config, moa_classes):
        """Test knowledge discovery initialization."""
        pathway_annotations = {'MoA_0': ['pathway1', 'pathway2']}
        
        discovery = KnowledgeDiscovery(
            model, config, moa_classes, pathway_annotations
        )
        
        assert discovery.model == model
        assert discovery.config == config
        assert discovery.moa_classes == moa_classes
        assert discovery.pathway_annotations == pathway_annotations
        assert discovery.significance_threshold == 0.05
    
    def test_discover_novel_associations(
        self, model, config, moa_classes, demo_compounds_list, demo_metadata
    ):
        """Test novel association discovery."""
        pathway_annotations = {
            'MoA_0': ['apoptosis', 'cell_cycle'],
            'MoA_1': ['metabolism', 'signaling']
        }
        
        discovery = KnowledgeDiscovery(
            model, config, moa_classes, pathway_annotations
        )
        
        results = discovery.discover_novel_associations(
            compound_data_list=demo_compounds_list,
            compound_metadata=demo_metadata,
            known_associations={}
        )
        
        # Check results structure
        assert 'total_compounds' in results
        assert 'association_methods' in results
        assert 'validated_associations' in results
        assert 'biological_hypotheses' in results
        assert 'discovery_statistics' in results
        
        # Check association methods
        methods = results['association_methods']
        assert 'statistical_associations' in methods
        assert 'clustering_associations' in methods
        assert 'pathway_enrichment' in methods
        assert 'novel_predictions' in methods
        
        # Check validated associations
        for assoc in results['validated_associations']:
            assert 'discovery_method' in assoc
            assert 'validation_score' in assoc
            assert 'confidence' in assoc
        
        # Check biological hypotheses
        for hypothesis in results['biological_hypotheses']:
            assert 'hypothesis_text' in hypothesis
            assert 'hypothesis_type' in hypothesis
            assert 'confidence' in hypothesis


class TestTherapeuticInsights:
    """Test therapeutic insights functionality."""
    
    def test_therapeutic_insights_initialization(self, model, config, moa_classes):
        """Test therapeutic insights initialization."""
        insights = TherapeuticInsights(model, config, moa_classes)
        
        assert insights.model == model
        assert insights.config == config
        assert insights.moa_classes == moa_classes
        assert insights.efficacy_threshold == 0.7
        assert insights.safety_threshold == 0.8
    
    def test_identify_therapeutic_targets(
        self, model, config, moa_classes, demo_compounds_list, demo_metadata
    ):
        """Test therapeutic target identification."""
        insights = TherapeuticInsights(model, config, moa_classes)
        
        disease_profile = {
            'disease_name': 'Test Disease',
            'target_pathways': ['apoptosis', 'metabolism'],
            'dysregulated_pathways': ['cell_cycle'],
            'biomarkers': ['marker1', 'marker2']
        }
        
        results = insights.identify_therapeutic_targets(
            disease_profile=disease_profile,
            compound_data_list=demo_compounds_list,
            compound_metadata=demo_metadata
        )
        
        # Check results structure
        assert 'disease_profile' in results
        assert 'target_analysis' in results
        assert 'promising_targets' in results
        assert 'druggability_assessment' in results
        assert 'therapeutic_recommendations' in results
        assert 'summary_metrics' in results
        
        # Check promising targets
        for target in results['promising_targets']:
            assert 'pathway' in target
            assert 'target_score' in target
            assert 'priority' in target
            assert 'druggable_compounds' in target
        
        # Check therapeutic recommendations
        for rec in results['therapeutic_recommendations']:
            assert 'target_pathway' in rec
            assert 'recommendation_type' in rec
            assert 'priority' in rec
            assert 'rationale' in rec
    
    def test_predict_drug_combinations(
        self, model, config, moa_classes, demo_compounds_list, demo_metadata
    ):
        """Test drug combination prediction."""
        insights = TherapeuticInsights(model, config, moa_classes)
        
        results = insights.predict_drug_combinations(
            compound_data_list=demo_compounds_list[:5],
            compound_metadata=demo_metadata[:5],
            target_disease="Test Disease"
        )
        
        # Check results structure
        assert 'target_disease' in results
        assert 'total_compounds' in results
        assert 'combination_scores' in results
        assert 'synergistic_combinations' in results
        assert 'safety_assessment' in results
        assert 'recommendations' in results
        assert 'combination_statistics' in results
        
        # Check combination scores
        for combo in results['combination_scores']:
            assert 'compound_1_index' in combo
            assert 'compound_2_index' in combo
            assert 'combination_score' in combo
            assert 'complementarity_score' in combo
            assert 'synergy_potential' in combo
        
        # Check synergistic combinations
        for combo in results['synergistic_combinations']:
            assert 'shared_moas' in combo
            assert 'synergy_mechanism' in combo
    
    def test_discover_biomarkers(
        self, model, config, moa_classes, demo_compounds_list, demo_metadata
    ):
        """Test biomarker discovery."""
        insights = TherapeuticInsights(model, config, moa_classes)
        
        results = insights.discover_biomarkers(
            compound_data_list=demo_compounds_list,
            compound_metadata=demo_metadata,
            clinical_outcomes=None
        )
        
        # Check results structure
        assert 'moa_outcome_analysis' in results
        assert 'predictive_moas' in results
        assert 'biomarker_hypotheses' in results
        assert 'validity_assessment' in results
        assert 'biomarker_statistics' in results
        
        # Check biomarker hypotheses
        for hypothesis in results['biomarker_hypotheses']:
            assert 'biomarker_type' in hypothesis
            assert 'hypothesis_text' in hypothesis
            assert 'clinical_utility' in hypothesis


class TestIntegration:
    """Test integration between interpretation and application components."""
    
    def test_end_to_end_workflow(
        self, model, config, moa_classes, demo_compounds_list, demo_metadata
    ):
        """Test complete end-to-end workflow."""
        # Initialize all components
        explainer = MoAExplainer(model, config, moa_classes)
        uncertainty_estimator = UncertaintyEstimator(model, config)
        repurposing_pipeline = DrugRepurposingPipeline(model, config, moa_classes)
        
        # Test workflow
        compound_data = demo_compounds_list[0]
        
        # 1. Explain prediction
        explanation = explainer.explain_prediction(compound_data)
        assert 'top_predictions' in explanation
        
        # 2. Estimate uncertainty
        uncertainty = uncertainty_estimator.estimate_uncertainty(compound_data, n_samples=5)
        assert 'total_uncertainty' in uncertainty
        
        # 3. Identify repurposing candidates
        repurposing_results = repurposing_pipeline.identify_repurposing_candidates(
            query_compound_data=compound_data,
            target_disease="Test Disease",
            candidate_compounds=demo_compounds_list[1:4],
            compound_metadata=demo_metadata[1:4]
        )
        assert 'ranked_candidates' in repurposing_results
        
        # Verify all components work together
        assert len(explanation['top_predictions']) > 0
        assert uncertainty['total_uncertainty'] >= 0
        assert len(repurposing_results['ranked_candidates']) > 0
    
    def test_report_generation(
        self, model, config, moa_classes, demo_compounds_list, demo_metadata
    ):
        """Test report generation functionality."""
        repurposing_pipeline = DrugRepurposingPipeline(model, config, moa_classes)
        
        # Get repurposing results
        results = repurposing_pipeline.identify_repurposing_candidates(
            query_compound_data=demo_compounds_list[0],
            target_disease="Test Disease",
            candidate_compounds=demo_compounds_list[1:3],
            compound_metadata=demo_metadata[1:3]
        )
        
        # Generate report
        with tempfile.TemporaryDirectory() as temp_dir:
            report_path = os.path.join(temp_dir, "test_report.txt")
            repurposing_pipeline.generate_repurposing_report(results, report_path)
            
            # Check report was created
            assert os.path.exists(report_path)
            
            # Check report content
            with open(report_path, 'r') as f:
                content = f.read()
                assert "DRUG REPURPOSING ANALYSIS REPORT" in content
                assert "Test Disease" in content
                assert "TOP REPURPOSING CANDIDATES" in content


if __name__ == "__main__":
    pytest.main([__file__])
