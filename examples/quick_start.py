#!/usr/bin/env python3
"""
Quick start example for MoA prediction framework.

This script demonstrates how to use the framework for basic MoA prediction tasks.
"""

import pandas as pd
from pathlib import Path
import sys

# Add the project root to the path
sys.path.append(str(Path(__file__).parent.parent))

from moa.utils.config import Config
from moa.data.processors import DataProcessor
from moa.data.validators import DataValidator


def main():
    """Quick start example."""
    print("MoA Prediction Framework - Quick Start Example")
    print("=" * 50)
    
    # 1. Load configuration
    print("\n1. Loading configuration...")
    config = Config("configs/config.yaml")
    print(f"   Prediction type: {config.get('scope.prediction_type')}")
    print(f"   Enabled modalities: {list(config.get('scope.modalities', {}).keys())}")
    
    # 2. Create sample data
    print("\n2. Creating sample dataset...")
    sample_data = pd.DataFrame({
        'molecule_chembl_id': [
            'CHEMBL25', 'CHEMBL521', 'CHEMBL113', 'CHEMBL1200766', 'CHEMBL154'
        ],
        'canonical_smiles': [
            'CCO',  # Ethanol
            'CC(=O)OC1=CC=CC=C1C(=O)O',  # Aspirin
            'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',  # Caffeine
            'CC(C)CC1=CC=C(C=C1)C(C)C(=O)O',  # Ibuprofen
            'CN(C)CCOC1=CC=C(C=C1)C(C2=CC=CC=C2)C3=CC=CC=C3'  # Diphenhydramine
        ],
        'mechanism_of_action': [
            'CNS depressant',
            'Cyclooxygenase inhibitor',
            'Adenosine receptor antagonist',
            'Cyclooxygenase inhibitor',
            'Histamine H1 receptor antagonist'
        ],
        'target_chembl_id': [
            'CHEMBL1', 'CHEMBL230', 'CHEMBL1824', 'CHEMBL230', 'CHEMBL231'
        ]
    })
    
    print(f"   Created dataset with {len(sample_data)} compounds")
    print("   Sample compounds:")
    for i, row in sample_data.iterrows():
        print(f"     {row['molecule_chembl_id']}: {row['mechanism_of_action']}")
    
    # 3. Initialize data processor
    print("\n3. Initializing data processor...")
    processor = DataProcessor(config)
    
    # 4. Process SMILES
    print("\n4. Processing SMILES...")
    processed_data = processor.smiles_processor.process_smiles_column(sample_data)
    print(f"   Processed {len(processed_data)} valid compounds")
    
    # Show SMILES processing results
    print("   SMILES processing results:")
    for i, row in processed_data.iterrows():
        original = row['canonical_smiles']
        standardized = row['standardized_smiles']
        print(f"     {original} -> {standardized}")
    
    # 5. Process labels
    print("\n5. Processing MoA labels...")
    processed_data = processor.label_processor.process_moa_labels(processed_data)
    
    # Show label processing results
    label_cols = [col for col in processed_data.columns if col.startswith('moa_')]
    print(f"   Created {len(label_cols)} label columns")
    
    if len(label_cols) > 0:
        print("   Label matrix:")
        print(processed_data[['molecule_chembl_id'] + label_cols[:5]].to_string(index=False))
    
    # 6. Validate data
    print("\n6. Validating processed data...")
    validator = DataValidator(config)
    
    # Run validation checks
    smiles_results = validator.validate_smiles(processed_data)
    label_results = validator.validate_labels(processed_data)
    size_results = validator.validate_dataset_size(processed_data)
    
    print("   Validation results:")
    all_results = {**smiles_results, **label_results, **size_results}
    for check, passed in all_results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"     {status}: {check}")
    
    # 7. Summary
    print("\n7. Summary")
    print("=" * 30)
    print(f"   Input compounds: {len(sample_data)}")
    print(f"   Valid compounds: {len(processed_data)}")
    print(f"   Unique MoAs: {len(processed_data['moa_cleaned'].unique())}")
    print(f"   Label columns: {len(label_cols)}")
    
    # Show MoA distribution
    moa_counts = processed_data['moa_cleaned'].value_counts()
    print("\n   MoA distribution:")
    for moa, count in moa_counts.items():
        print(f"     {moa}: {count}")
    
    print("\n✓ Quick start example completed successfully!")
    print("\nNext steps:")
    print("  1. Run full data collection: python scripts/download_data.py")
    print("  2. Process complete dataset: python scripts/process_data.py --create-splits")
    print("  3. Explore data: jupyter notebook notebooks/01_data_exploration.ipynb")
    print("  4. Train models: moa-train --config configs/config.yaml")


if __name__ == "__main__":
    main()
