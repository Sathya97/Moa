"""Tests for data processing functionality."""

import pytest
import pandas as pd
import tempfile
from pathlib import Path

from moa.utils.config import Config
from moa.data.processors import SMILESProcessor, DuplicateHandler, LabelProcessor, DataProcessor
from moa.data.validators import DataValidator


@pytest.fixture
def sample_config():
    """Create a sample configuration for testing."""
    config_dict = {
        "processing": {
            "smiles": {
                "standardizer": "rdkit",
                "remove_salts": True,
                "neutralize": True,
                "canonicalize": True
            },
            "duplicates": {
                "method": "smiles",
                "keep": "first"
            },
            "quality": {
                "max_molecular_weight": 1000,
                "min_molecular_weight": 150,
                "max_heavy_atoms": 100,
                "min_heavy_atoms": 5
            }
        },
        "scope": {
            "prediction_type": "multi_label"
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
def sample_data():
    """Create sample data for testing."""
    return pd.DataFrame({
        'molecule_chembl_id': ['CHEMBL1', 'CHEMBL2', 'CHEMBL3', 'CHEMBL4'],
        'canonical_smiles': ['CCO', 'CC(=O)OC1=CC=CC=C1C(=O)O', 'CCO', 'invalid_smiles'],
        'mechanism_of_action': ['CNS depressant', 'Cyclooxygenase inhibitor', 'CNS depressant', 'Unknown'],
        'target_chembl_id': ['CHEMBL1', 'CHEMBL230', 'CHEMBL1', 'CHEMBL999']
    })


class TestSMILESProcessor:
    """Test SMILES processing functionality."""
    
    def test_standardize_smiles(self, sample_config):
        """Test SMILES standardization."""
        processor = SMILESProcessor(sample_config)
        
        # Test valid SMILES
        result = processor.standardize_smiles('CCO')
        assert result == 'CCO'
        
        # Test invalid SMILES
        result = processor.standardize_smiles('invalid')
        assert result is None
    
    def test_validate_smiles(self, sample_config):
        """Test SMILES validation."""
        processor = SMILESProcessor(sample_config)
        
        # Test valid SMILES
        assert processor.validate_smiles('CCO') == True
        
        # Test invalid SMILES
        assert processor.validate_smiles('invalid') == False
    
    def test_process_smiles_column(self, sample_config, sample_data):
        """Test processing SMILES column in DataFrame."""
        processor = SMILESProcessor(sample_config)
        
        result = processor.process_smiles_column(sample_data)
        
        # Should have standardized_smiles column
        assert 'standardized_smiles' in result.columns
        
        # Should remove invalid SMILES
        assert len(result) < len(sample_data)
        
        # All remaining SMILES should be valid
        assert result['standardized_smiles'].notna().all()


class TestDuplicateHandler:
    """Test duplicate handling functionality."""
    
    def test_remove_duplicates(self, sample_config, sample_data):
        """Test duplicate removal."""
        handler = DuplicateHandler(sample_config)
        
        # Add standardized_smiles column
        sample_data['standardized_smiles'] = sample_data['canonical_smiles']
        
        result = handler.remove_duplicates(sample_data)
        
        # Should remove duplicates
        assert len(result) < len(sample_data)
        
        # Should not have duplicate SMILES
        assert not result['standardized_smiles'].duplicated().any()


class TestLabelProcessor:
    """Test label processing functionality."""
    
    def test_process_moa_labels_multilabel(self, sample_config, sample_data):
        """Test multi-label MoA processing."""
        processor = LabelProcessor(sample_config)
        
        result = processor.process_moa_labels(sample_data)
        
        # Should have cleaned MoA column
        assert 'moa_cleaned' in result.columns
        
        # Should have label columns for multi-label
        label_cols = [col for col in result.columns if col.startswith('moa_')]
        assert len(label_cols) > 0
    
    def test_process_moa_labels_singlelabel(self, sample_config, sample_data):
        """Test single-label MoA processing."""
        # Change config to single label
        sample_config.set('scope.prediction_type', 'single_label')
        processor = LabelProcessor(sample_config)
        
        result = processor.process_moa_labels(sample_data)
        
        # Should have primary MoA column
        assert 'moa_primary' in result.columns


class TestDataValidator:
    """Test data validation functionality."""
    
    def test_validate_smiles(self, sample_config):
        """Test SMILES validation."""
        validator = DataValidator(sample_config)
        
        # Create test data
        test_data = pd.DataFrame({
            'standardized_smiles': ['CCO', 'CC(=O)O', 'CCO']  # One duplicate
        })
        
        results = validator.validate_smiles(test_data)
        
        assert 'smiles_column_exists' in results
        assert 'no_missing_smiles' in results
        assert 'all_smiles_valid' in results
        assert 'no_duplicate_smiles' in results
        
        # Should detect duplicate
        assert results['no_duplicate_smiles'] == False
    
    def test_validate_labels(self, sample_config):
        """Test label validation."""
        validator = DataValidator(sample_config)
        
        # Create test data with label columns
        test_data = pd.DataFrame({
            'moa_label1': [1, 0, 1],
            'moa_label2': [0, 1, 0]
        })
        
        results = validator.validate_labels(test_data)
        
        assert 'label_columns_exist' in results
        assert 'no_missing_labels' in results
        assert 'no_empty_labels' in results


class TestDataProcessor:
    """Test integrated data processing."""
    
    def test_process_chembl_data(self, sample_config):
        """Test ChEMBL data processing."""
        processor = DataProcessor(sample_config)
        
        # Create temporary input directory with sample data
        with tempfile.TemporaryDirectory() as temp_dir:
            input_dir = Path(temp_dir)
            chembl_dir = input_dir / "chembl"
            chembl_dir.mkdir()
            
            # Create sample mechanism and compound files
            mechanisms = pd.DataFrame({
                'molecule_chembl_id': ['CHEMBL1', 'CHEMBL2'],
                'mechanism_of_action': ['CNS depressant', 'Cyclooxygenase inhibitor'],
                'target_chembl_id': ['CHEMBL1', 'CHEMBL230']
            })
            
            compounds = pd.DataFrame({
                'molecule_chembl_id': ['CHEMBL1', 'CHEMBL2'],
                'canonical_smiles': ['CCO', 'CC(=O)OC1=CC=CC=C1C(=O)O']
            })
            
            mechanisms.to_csv(chembl_dir / "mechanisms.csv", index=False)
            compounds.to_csv(chembl_dir / "compounds.csv", index=False)
            
            # Process data
            result = processor.process_chembl_data(input_dir)
            
            # Should return processed DataFrame
            assert isinstance(result, pd.DataFrame)
            assert len(result) > 0
            assert 'standardized_smiles' in result.columns


def test_integration_pipeline(sample_config):
    """Test the complete data processing pipeline."""
    processor = DataProcessor(sample_config)
    
    # Create sample data
    sample_data = pd.DataFrame({
        'molecule_chembl_id': ['CHEMBL1', 'CHEMBL2', 'CHEMBL3'],
        'canonical_smiles': ['CCO', 'CC(=O)O', 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C'],
        'mechanism_of_action': ['CNS depressant', 'Metabolic', 'Adenosine receptor antagonist']
    })
    
    # Process SMILES
    processed = processor.smiles_processor.process_smiles_column(sample_data)
    
    # Remove duplicates
    processed = processor.duplicate_handler.remove_duplicates(processed)
    
    # Process labels
    processed = processor.label_processor.process_moa_labels(processed)
    
    # Validate final result
    validator = DataValidator(sample_config)
    smiles_results = validator.validate_smiles(processed)
    label_results = validator.validate_labels(processed)
    
    # Should pass basic validations
    assert smiles_results['smiles_column_exists']
    assert smiles_results['all_smiles_valid']
    assert label_results['label_columns_exist']


if __name__ == "__main__":
    pytest.main([__file__])
