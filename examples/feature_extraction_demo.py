#!/usr/bin/env python3
"""
Feature extraction demonstration for MoA prediction framework.

This script demonstrates the novel feature engineering approaches including:
- Chemical graph features with counterfactual analysis
- Mechanism Tokens (MechTokens)
- Perturbational biology features
- Protein pocket features (optional)
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add the project root to the path
sys.path.append(str(Path(__file__).parent.parent))

from moa.utils.config import Config
from moa.features.feature_extractor import MultiModalFeatureExtractor
from moa.data.processors import DataProcessor


def create_sample_data():
    """Create sample data for demonstration."""
    print("Creating sample dataset...")
    
    # Sample compounds with known MoAs
    sample_data = pd.DataFrame({
        'molecule_chembl_id': [
            'CHEMBL25', 'CHEMBL521', 'CHEMBL113', 'CHEMBL1200766', 'CHEMBL154',
            'CHEMBL6', 'CHEMBL1201585', 'CHEMBL744', 'CHEMBL1200960', 'CHEMBL1201'
        ],
        'canonical_smiles': [
            'CCO',  # Ethanol
            'CC(=O)OC1=CC=CC=C1C(=O)O',  # Aspirin
            'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',  # Caffeine
            'CC(C)CC1=CC=C(C=C1)C(C)C(=O)O',  # Ibuprofen
            'CN(C)CCOC1=CC=C(C=C1)C(C2=CC=CC=C2)C3=CC=CC=C3',  # Diphenhydramine
            'CC(C)(C)NCC(C1=CC(=C(C=C1)O)CO)O',  # Salbutamol
            'CN1CCN(CC1)C2=C(C=C3C(=C2)N=CN=C3NC4=CC=C(C=C4)OC)F',  # Gefitinib
            'CC1=C(C=C(C=C1)C(=O)C2=CC=CC=C2)C',  # Tolmetin
            'COC1=CC=C(C=C1)C2=CC(=NN2C3=CC=C(C=C3)S(=O)(=O)N)C(F)(F)F',  # Celecoxib
            'CN(C)CCCN1C2=CC=CC=C2SC3=C1C=C(C=C3)Cl'  # Chlorpromazine
        ],
        'mechanism_of_action': [
            'CNS depressant',
            'Cyclooxygenase inhibitor',
            'Adenosine receptor antagonist',
            'Cyclooxygenase inhibitor',
            'Histamine H1 receptor antagonist',
            'Beta-2 adrenergic receptor agonist',
            'EGFR tyrosine kinase inhibitor',
            'Cyclooxygenase inhibitor',
            'Cyclooxygenase-2 inhibitor',
            'Dopamine receptor antagonist'
        ],
        'target_chembl_id': [
            'CHEMBL1', 'CHEMBL230', 'CHEMBL1824', 'CHEMBL230', 'CHEMBL231',
            'CHEMBL210', 'CHEMBL203', 'CHEMBL230', 'CHEMBL230', 'CHEMBL217'
        ]
    })
    
    return sample_data


def create_sample_data_sources():
    """Create sample data sources for mechanism tokens."""
    print("Creating sample data sources...")
    
    # Sample mechanisms data
    mechanisms_df = pd.DataFrame({
        'molecule_chembl_id': ['CHEMBL25', 'CHEMBL521', 'CHEMBL113', 'CHEMBL1200766'],
        'target_chembl_id': ['CHEMBL1', 'CHEMBL230', 'CHEMBL1824', 'CHEMBL230'],
        'mechanism_of_action': ['CNS depressant', 'Cyclooxygenase inhibitor', 
                               'Adenosine receptor antagonist', 'Cyclooxygenase inhibitor']
    })
    
    # Sample targets data
    targets_df = pd.DataFrame({
        'target_chembl_id': ['CHEMBL1', 'CHEMBL230', 'CHEMBL1824', 'CHEMBL210'],
        'pref_name': ['Alcohol dehydrogenase', 'Cyclooxygenase-1', 
                     'Adenosine A2a receptor', 'Beta-2 adrenergic receptor'],
        'target_type': ['PROTEIN', 'PROTEIN', 'PROTEIN', 'PROTEIN']
    })
    
    # Sample pathways data
    pathways_df = pd.DataFrame({
        'stId': ['R-HSA-1234', 'R-HSA-5678', 'R-HSA-9012'],
        'displayName': ['Alcohol metabolism', 'Arachidonic acid metabolism', 'Adenosine signaling']
    })
    
    # Sample protein-pathway mappings
    protein_pathways_df = pd.DataFrame({
        'uniprot_id': ['P00325', 'P23219', 'P29274'],
        'pathway_id': ['R-HSA-1234', 'R-HSA-5678', 'R-HSA-9012'],
        'pathway_name': ['Alcohol metabolism', 'Arachidonic acid metabolism', 'Adenosine signaling']
    })
    
    # Sample LINCS signatures (simplified)
    lincs_signatures = pd.DataFrame({
        'sig_id': [f'SIG_{i:04d}' for i in range(20)],
        'pert_iname': ['compound_25', 'compound_521'] * 10,
        'cell_id': ['MCF7', 'PC3'] * 10,
        'pert_time': ['24h'] * 20,
        'pert_dose': ['10 µM'] * 20
    })
    
    return {
        'mechanisms': mechanisms_df,
        'targets': targets_df,
        'pathways': pathways_df,
        'protein_pathways': protein_pathways_df,
        'lincs_signatures': lincs_signatures
    }


def main():
    """Main demonstration function."""
    print("MoA Prediction Framework - Feature Extraction Demo")
    print("=" * 60)
    
    # 1. Load configuration
    print("\n1. Loading configuration...")
    config = Config("configs/config.yaml")
    
    # Enable all modalities for demonstration
    config.set("scope.modalities.chemistry", True)
    config.set("scope.modalities.targets", True)
    config.set("scope.modalities.pathways", True)
    config.set("scope.modalities.perturbation", True)
    config.set("scope.modalities.structures", False)  # Keep optional for demo
    
    print(f"   Enabled modalities: {config.get('scope.modalities')}")
    
    # 2. Create sample data
    print("\n2. Creating sample data...")
    sample_data = create_sample_data()
    data_sources = create_sample_data_sources()
    
    print(f"   Created dataset with {len(sample_data)} compounds")
    print("   Sample compounds:")
    for i, row in sample_data.head().iterrows():
        print(f"     {row['molecule_chembl_id']}: {row['mechanism_of_action']}")
    
    # 3. Process data
    print("\n3. Processing compound data...")
    processor = DataProcessor(config)
    
    # Process SMILES
    processed_data = processor.smiles_processor.process_smiles_column(sample_data)
    print(f"   Processed {len(processed_data)} valid compounds")
    
    # Process labels
    processed_data = processor.label_processor.process_moa_labels(processed_data)
    label_cols = [col for col in processed_data.columns if col.startswith('moa_')]
    print(f"   Created {len(label_cols)} MoA labels")
    
    # 4. Initialize feature extractor
    print("\n4. Initializing multi-modal feature extractor...")
    feature_extractor = MultiModalFeatureExtractor(config)
    print(f"   Initialized extractors: {list(feature_extractor.extractors.keys())}")
    
    # 5. Extract features
    print("\n5. Extracting multi-modal features...")
    print("   This may take a few minutes...")
    
    try:
        features = feature_extractor.extract_all_features(processed_data, data_sources)
        
        print("\n   Feature extraction completed!")
        print("   Extracted features:")
        
        for modality, modality_features in features.items():
            print(f"\n   {modality.upper()} Features:")
            
            if modality == "chemistry":
                if "molecular_graphs" in modality_features:
                    graphs = modality_features["molecular_graphs"]
                    valid_graphs = [g for g in graphs if g is not None]
                    print(f"     - Molecular graphs: {len(valid_graphs)}/{len(graphs)} valid")
                
                if "molecular_descriptors" in modality_features:
                    descriptors = modality_features["molecular_descriptors"]
                    print(f"     - Molecular descriptors: {len(descriptors)} compounds")
                    if descriptors:
                        print(f"       Example features: {list(descriptors[0].keys())[:5]}...")
                
                if "counterfactual_scores" in modality_features:
                    cf_scores = modality_features["counterfactual_scores"]
                    print(f"     - Counterfactual scores: {len(cf_scores)} MoAs")
                    for moa, scores in list(cf_scores.items())[:2]:
                        print(f"       {moa}: {len(scores)} fragments analyzed")
            
            elif modality == "mechanism_tokens":
                if "compound_tokens" in modality_features:
                    tokens = modality_features["compound_tokens"]
                    print(f"     - Compound tokens: {len(tokens)} compounds")
                    if tokens:
                        token_dim = list(tokens.values())[0].shape[0]
                        print(f"       Token dimension: {token_dim}")
                
                if "moa_tokens" in modality_features:
                    moa_tokens = modality_features["moa_tokens"]
                    print(f"     - MoA tokens: {len(moa_tokens)} MoAs")
            
            elif modality == "perturbation":
                if "meta_signatures" in modality_features:
                    meta_sigs = modality_features["meta_signatures"]
                    print(f"     - Meta-signatures: {len(meta_sigs)} compounds")
                    if meta_sigs:
                        sig_dim = list(meta_sigs.values())[0].shape[0]
                        print(f"       Signature dimension: {sig_dim}")
                
                if "pathway_scores" in modality_features:
                    pathway_scores = modality_features["pathway_scores"]
                    print(f"     - Pathway scores: {len(pathway_scores)} compounds")
                    if "pathway_names" in modality_features:
                        n_pathways = len(modality_features["pathway_names"])
                        print(f"       Number of pathways: {n_pathways}")
        
        # 6. Save features
        print("\n6. Saving extracted features...")
        save_dir = Path("data/features")
        feature_extractor.save_features(features, save_dir)
        print(f"   Features saved to {save_dir}")
        
        # 7. Demonstrate feature loading
        print("\n7. Testing feature loading...")
        loaded_features = feature_extractor.load_features(save_dir)
        print(f"   Loaded {len(loaded_features)} modalities")
        
        # 8. Summary
        print("\n8. Summary")
        print("=" * 40)
        print(f"   Input compounds: {len(sample_data)}")
        print(f"   Processed compounds: {len(processed_data)}")
        print(f"   Extracted modalities: {len(features)}")
        
        # Feature dimensions summary
        print("\n   Feature dimensions:")
        for modality, modality_features in features.items():
            if modality == "chemistry" and "molecular_descriptors" in modality_features:
                desc_example = modality_features["molecular_descriptors"][0]
                n_features = sum(1 if not isinstance(v, np.ndarray) else len(v) for v in desc_example.values())
                print(f"     Chemical descriptors: ~{n_features} features")
            
            elif modality == "mechanism_tokens" and "compound_tokens" in modality_features:
                tokens = modality_features["compound_tokens"]
                if tokens:
                    token_dim = list(tokens.values())[0].shape[0]
                    print(f"     Mechanism tokens: {token_dim} dimensions")
            
            elif modality == "perturbation":
                if "meta_signatures" in modality_features:
                    meta_sigs = modality_features["meta_signatures"]
                    if meta_sigs:
                        sig_dim = list(meta_sigs.values())[0].shape[0]
                        print(f"     Gene signatures: {sig_dim} genes")
                
                if "pathway_scores" in modality_features:
                    pathway_scores = modality_features["pathway_scores"]
                    if pathway_scores:
                        pathway_dim = list(pathway_scores.values())[0].shape[0]
                        print(f"     Pathway scores: {pathway_dim} pathways")
        
        print("\n✓ Feature extraction demo completed successfully!")
        print("\nNext steps:")
        print("  1. Implement model architectures (Phase 3)")
        print("  2. Train multi-modal models with these features")
        print("  3. Evaluate on benchmark datasets")
        
    except Exception as e:
        print(f"\n✗ Feature extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
