"""Tests for feature extraction functionality."""

import pytest
import pandas as pd
import numpy as np
import tempfile
from pathlib import Path

from moa.utils.config import Config
from moa.features.chemical import ChemicalFeatureExtractor, MolecularGraphFeaturizer
from moa.features.mechanism_tokens import MechTokenFeatureExtractor
from moa.features.perturbational import PerturbationalFeatureExtractor
from moa.features.feature_extractor import MultiModalFeatureExtractor


@pytest.fixture
def sample_config():
    """Create a sample configuration for testing."""
    config_dict = {
        "scope": {
            "modalities": {
                "chemistry": True,
                "targets": True,
                "pathways": True,
                "perturbation": False,
                "structures": False
            }
        },
        "features": {
            "chemistry": {
                "molecular_descriptors": ["morgan_fingerprints", "rdkit_descriptors"],
                "graph_features": {
                    "node_features": ["atomic_number", "formal_charge", "aromatic"],
                    "edge_features": ["bond_type", "conjugated"]
                },
                "substructure_analysis": {
                    "enable_counterfactual": True,
                    "fragment_size_range": [3, 6],
                    "max_fragments": 100
                }
            },
            "mechanism_tokens": {
                "ontology_sources": ["chembl"],
                "embedding_dim": 128,
                "node2vec_params": {
                    "dimensions": 64,
                    "walk_length": 20,
                    "num_walks": 5
                }
            }
        }
    }
    
    # Create temporary config
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        import yaml
        yaml.dump(config_dict, f)
        config_path = f.name
    
    config = Config(config_path)
    yield config
    
    # Cleanup
    Path(config_path).unlink()


@pytest.fixture
def sample_compound_data():
    """Create sample compound data for testing."""
    return pd.DataFrame({
        'molecule_chembl_id': ['CHEMBL1', 'CHEMBL2', 'CHEMBL3'],
        'canonical_smiles': ['CCO', 'CC(=O)O', 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C'],
        'standardized_smiles': ['CCO', 'CC(=O)O', 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C'],
        'mechanism_of_action': ['CNS depressant', 'Metabolic', 'Adenosine receptor antagonist'],
        'moa_cleaned': ['cns depressant', 'metabolic', 'adenosine receptor antagonist'],
        'moa_list': [['cns depressant'], ['metabolic'], ['adenosine receptor antagonist']],
        'target_chembl_id': ['CHEMBL1', 'CHEMBL2', 'CHEMBL3']
    })


@pytest.fixture
def sample_data_sources():
    """Create sample data sources for testing."""
    mechanisms_df = pd.DataFrame({
        'molecule_chembl_id': ['CHEMBL1', 'CHEMBL2', 'CHEMBL3'],
        'target_chembl_id': ['CHEMBL1', 'CHEMBL2', 'CHEMBL3'],
        'mechanism_of_action': ['CNS depressant', 'Metabolic', 'Adenosine receptor antagonist']
    })
    
    targets_df = pd.DataFrame({
        'target_chembl_id': ['CHEMBL1', 'CHEMBL2', 'CHEMBL3'],
        'pref_name': ['Target 1', 'Target 2', 'Target 3'],
        'target_type': ['PROTEIN', 'PROTEIN', 'PROTEIN']
    })
    
    return {
        'mechanisms': mechanisms_df,
        'targets': targets_df
    }


class TestMolecularGraphFeaturizer:
    """Test molecular graph featurization."""
    
    def test_smiles_to_graph(self, sample_config):
        """Test SMILES to graph conversion."""
        featurizer = MolecularGraphFeaturizer(sample_config)
        
        # Test valid SMILES
        graph = featurizer.smiles_to_graph('CCO')
        assert graph is not None
        assert graph.x.shape[0] > 0  # Should have nodes
        assert graph.edge_index.shape[0] == 2  # Should have edges
        
        # Test invalid SMILES
        graph = featurizer.smiles_to_graph('invalid')
        assert graph is None
    
    def test_batch_conversion(self, sample_config):
        """Test batch SMILES to graph conversion."""
        featurizer = MolecularGraphFeaturizer(sample_config)
        
        smiles_list = ['CCO', 'CC(=O)O', 'invalid']
        graphs = featurizer.batch_smiles_to_graphs(smiles_list)
        
        assert len(graphs) == 3
        assert graphs[0] is not None
        assert graphs[1] is not None
        assert graphs[2] is None


class TestChemicalFeatureExtractor:
    """Test chemical feature extraction."""
    
    def test_extract_features(self, sample_config, sample_compound_data):
        """Test chemical feature extraction."""
        extractor = ChemicalFeatureExtractor(sample_config)
        
        smiles_list = sample_compound_data['standardized_smiles'].tolist()
        
        # Extract features without labels (no counterfactual analysis)
        features = extractor.extract_features(smiles_list)
        
        assert 'molecular_graphs' in features
        assert 'molecular_descriptors' in features
        assert len(features['molecular_graphs']) == len(smiles_list)
        assert len(features['molecular_descriptors']) == len(smiles_list)
    
    def test_extract_features_with_labels(self, sample_config, sample_compound_data):
        """Test chemical feature extraction with labels for counterfactual analysis."""
        extractor = ChemicalFeatureExtractor(sample_config)
        
        smiles_list = sample_compound_data['standardized_smiles'].tolist()
        
        # Create simple binary labels
        labels = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        moa_names = ['moa1', 'moa2', 'moa3']
        
        features = extractor.extract_features(smiles_list, labels, moa_names)
        
        assert 'molecular_graphs' in features
        assert 'molecular_descriptors' in features
        assert 'counterfactual_scores' in features
        assert 'causal_fragments' in features


class TestMechTokenFeatureExtractor:
    """Test mechanism token feature extraction."""
    
    def test_build_mechanism_tokens(self, sample_config, sample_data_sources):
        """Test mechanism token building."""
        extractor = MechTokenFeatureExtractor(sample_config)
        
        # Build mechanism tokens
        extractor.build_mechanism_tokens(sample_data_sources)
        
        assert extractor.ontology_graph is not None
        assert extractor.mechanism_tokens is not None
        assert len(extractor.mechanism_tokens) > 0
    
    def test_extract_compound_tokens(self, sample_config, sample_data_sources):
        """Test compound token extraction."""
        extractor = MechTokenFeatureExtractor(sample_config)
        
        # Build mechanism tokens first
        extractor.build_mechanism_tokens(sample_data_sources)
        
        # Extract compound tokens
        compound_ids = ['CHEMBL1', 'CHEMBL2', 'CHEMBL3']
        compound_tokens = extractor.extract_compound_tokens(compound_ids)
        
        assert len(compound_tokens) == len(compound_ids)
        for compound_id in compound_ids:
            assert compound_id in compound_tokens
            assert isinstance(compound_tokens[compound_id], np.ndarray)


class TestPerturbationalFeatureExtractor:
    """Test perturbational feature extraction."""
    
    def test_extract_perturbational_features(self, sample_config):
        """Test perturbational feature extraction."""
        extractor = PerturbationalFeatureExtractor(sample_config)
        
        # Create sample LINCS signatures
        signatures_df = pd.DataFrame({
            'sig_id': ['SIG_001', 'SIG_002', 'SIG_003'],
            'pert_iname': ['compound_1', 'compound_2', 'compound_1'],
            'cell_id': ['MCF7', 'MCF7', 'PC3'],
            'pert_time': ['24h', '24h', '24h'],
            'pert_dose': ['10 µM', '10 µM', '10 µM']
        })
        
        compound_names = ['compound_1', 'compound_2']
        
        features = extractor.extract_perturbational_features(signatures_df, compound_names)
        
        assert 'meta_signatures' in features
        assert 'pathway_scores' in features
        assert len(features['meta_signatures']) == len(compound_names)
        assert len(features['pathway_scores']) == len(compound_names)


class TestMultiModalFeatureExtractor:
    """Test multi-modal feature extraction."""
    
    def test_initialization(self, sample_config):
        """Test multi-modal feature extractor initialization."""
        extractor = MultiModalFeatureExtractor(sample_config)
        
        # Should initialize extractors based on enabled modalities
        assert 'chemistry' in extractor.extractors
        assert 'mechanism_tokens' in extractor.extractors
        # perturbation and structures should be disabled in sample config
    
    def test_extract_all_features(self, sample_config, sample_compound_data, sample_data_sources):
        """Test extraction of all features."""
        extractor = MultiModalFeatureExtractor(sample_config)
        
        features = extractor.extract_all_features(sample_compound_data, sample_data_sources)
        
        # Should have features from enabled modalities
        assert 'chemistry' in features
        assert 'mechanism_tokens' in features
        
        # Chemistry features
        chemistry_features = features['chemistry']
        assert 'molecular_graphs' in chemistry_features
        assert 'molecular_descriptors' in chemistry_features
        
        # Mechanism token features
        token_features = features['mechanism_tokens']
        assert 'compound_tokens' in token_features
    
    def test_save_and_load_features(self, sample_config, sample_compound_data, sample_data_sources):
        """Test saving and loading features."""
        extractor = MultiModalFeatureExtractor(sample_config)
        
        # Extract features
        features = extractor.extract_all_features(sample_compound_data, sample_data_sources)
        
        # Save features
        with tempfile.TemporaryDirectory() as temp_dir:
            save_dir = Path(temp_dir)
            extractor.save_features(features, save_dir)
            
            # Load features
            loaded_features = extractor.load_features(save_dir)
            
            # Check that loaded features match original
            assert set(loaded_features.keys()) == set(features.keys())
            
            for modality in features.keys():
                assert modality in loaded_features


def test_integration_feature_extraction(sample_config, sample_compound_data, sample_data_sources):
    """Test the complete feature extraction pipeline."""
    extractor = MultiModalFeatureExtractor(sample_config)
    
    # Extract all features
    features = extractor.extract_all_features(sample_compound_data, sample_data_sources)
    
    # Verify we have the expected structure
    assert isinstance(features, dict)
    assert len(features) > 0
    
    # Verify chemistry features
    if 'chemistry' in features:
        chemistry = features['chemistry']
        assert 'molecular_graphs' in chemistry
        assert 'molecular_descriptors' in chemistry
        
        # Check that we have features for all compounds
        n_compounds = len(sample_compound_data)
        assert len(chemistry['molecular_graphs']) == n_compounds
        assert len(chemistry['molecular_descriptors']) == n_compounds
    
    # Verify mechanism tokens
    if 'mechanism_tokens' in features:
        tokens = features['mechanism_tokens']
        assert 'compound_tokens' in tokens
        
        # Check that we have tokens for all compounds
        compound_ids = sample_compound_data['molecule_chembl_id'].tolist()
        for compound_id in compound_ids:
            assert compound_id in tokens['compound_tokens']


if __name__ == "__main__":
    pytest.main([__file__])
