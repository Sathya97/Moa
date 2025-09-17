# Phase 2: Feature Engineering - Novel Representations

## Overview

Phase 2 implements cutting-edge feature engineering techniques for multi-modal MoA prediction, including chemical graph features with counterfactual analysis, mechanism tokens, perturbational biology features, and protein pocket representations.

## 🎯 Objectives

- Implement molecular graph features with counterfactual fragment analysis
- Build Mechanism Tokens (MechTokens) for ontology-aware embeddings
- Generate perturbational biology features from LINCS L1000 data
- Create protein pocket features using 3D structural information

## 📁 Components

### 1. Chemical Graph Features with Counterfactual Analysis
- **File**: `moa/features/chemical.py`
- **Novel Contribution**: Substructure counterfactual analysis for causal fragment identification
- **Output**: Rich molecular graphs with fragment importance scores

### 2. Mechanism Tokens (MechTokens)
- **File**: `moa/features/mechanism_tokens.py`
- **Novel Contribution**: Ontology-aware embeddings encoding drug-target-pathway relationships
- **Output**: 128-dimensional mechanism embeddings

### 3. Perturbational Biology Features
- **File**: `moa/features/perturbational.py`
- **Purpose**: LINCS L1000 gene expression signatures and pathway activity scores
- **Output**: 978 gene features + 50 pathway scores

### 4. Protein Structure Features
- **File**: `moa/features/protein_structure.py`
- **Purpose**: 3D binding site representations using AlphaFold structures
- **Output**: 256-dimensional pocket features

## 🚀 Execution Instructions

### Prerequisites
```bash
# Install additional dependencies for feature engineering
pip install torch-geometric networkx node2vec scikit-learn
pip install biotite  # For protein structure processing

# Download demo datasets (if not already done)
python moa/data/download_demo_data.py
```

### Step 1: Run Feature Engineering Demo
```bash
# Execute complete Phase 2 workflow
python examples/phase2_feature_demo.py

# Expected runtime: 15-20 minutes
# Memory usage: ~4GB
# GPU recommended for faster processing
```

### Step 2: Interactive Feature Exploration
```bash
# Launch feature engineering notebook
jupyter notebook notebooks/02_feature_engineering.ipynb

# Explore different feature types interactively
```

### Step 3: Custom Feature Configuration
```bash
# Modify feature settings
vim configs/config.yaml

# Key sections:
# - features.chemical.counterfactual_analysis
# - features.mechanism_tokens.embedding_dim
# - features.perturbational.pathway_databases
```

## 📊 Expected Results

### 1. Chemical Graph Features
```
🧪 Chemical Feature Extraction Results:
├── Compounds processed: ~40,000
├── Average atoms per molecule: 25-30
├── Average bonds per molecule: 28-35
├── Node features per atom: 64 dimensions
├── Edge features per bond: 16 dimensions
└── Counterfactual fragments identified: ~15 per molecule
```

**Feature Breakdown:**
- **Node Features**: Atomic number, hybridization, formal charge, aromaticity, etc.
- **Edge Features**: Bond type, conjugation, ring membership, stereochemistry
- **Graph Properties**: Molecular weight, LogP, TPSA, rotatable bonds
- **Counterfactual Weights**: Fragment importance scores (0-1)

### 2. Mechanism Tokens (MechTokens)
```
🔗 MechToken Generation Results:
├── Drug-target pairs: ~15,000
├── Target-pathway mappings: ~8,000
├── Biological ontology nodes: ~5,000
├── Embedding dimensions: 128
├── Node2vec walks: 1,000 per node
└── Training epochs: 100
```

**MechToken Components:**
- **Drug Embeddings**: Chemical structure-aware representations
- **Target Embeddings**: Protein function and family information
- **Pathway Embeddings**: Biological process and molecular function
- **Hierarchical Encoding**: Multi-level biological organization

### 3. Perturbational Biology Features
```
🧬 Perturbational Feature Results:
├── LINCS L1000 signatures: ~1,000 per compound
├── Gene expression features: 978 landmark genes
├── Pathway databases: KEGG, Reactome, GO
├── Pathway activity scores: 50 major pathways
├── Meta-signature generation: Consensus across cell lines
└── GSVA enrichment scores: Normalized (-1 to +1)
```

**Feature Types:**
- **Gene Expression**: Z-scored differential expression
- **Pathway Scores**: GSVA/ssGSEA enrichment scores
- **Meta-signatures**: Consensus across multiple cell lines
- **Perturbation Strength**: Magnitude of biological response

### 4. Protein Structure Features
```
🏗️ Protein Structure Feature Results:
├── AlphaFold structures: ~2,000 targets
├── Binding sites identified: ~1,500
├── Pocket descriptors: 256 dimensions
├── 3D coordinates: Cartesian (x,y,z)
├── Physicochemical properties: Hydrophobicity, electrostatics
└── Geometric features: Volume, surface area, shape
```

**Pocket Features:**
- **Geometric**: Volume, surface area, depth, shape descriptors
- **Physicochemical**: Hydrophobicity, charge distribution, H-bonding
- **Evolutionary**: Conservation scores, family-specific patterns
- **Druggability**: Pocket druggability scores

### 5. Generated Feature Files
```
outputs/
├── features/
│   ├── chemical_graphs.pkl           # Molecular graphs with counterfactuals
│   ├── mechanism_tokens.pkl          # MechToken embeddings
│   ├── perturbational_features.pkl   # LINCS L1000 signatures
│   ├── protein_pockets.pkl           # 3D binding site features
│   └── feature_statistics.json      # Comprehensive statistics
├── visualizations/
│   ├── molecular_graphs/             # Graph visualizations
│   ├── mechanism_networks/           # MechToken networks
│   ├── pathway_heatmaps/            # Perturbational heatmaps
│   └── pocket_structures/           # 3D pocket visualizations
└── reports/
    └── phase2_feature_report.txt    # Detailed feature analysis
```

## 🔍 Key Outputs Explained

### 1. Chemical Graph Features
```python
# Example chemical feature structure
chemical_features = {
    'node_features': torch.tensor([n_atoms, 64]),      # Atomic features
    'edge_index': torch.tensor([2, n_bonds]),          # Bond connectivity
    'edge_features': torch.tensor([n_bonds, 16]),      # Bond features
    'batch': torch.tensor([n_atoms]),                  # Batch assignment
    'counterfactual_weights': torch.tensor([n_atoms])  # Fragment importance
}
```

### 2. Mechanism Tokens
```python
# Example MechToken structure
mechanism_tokens = {
    'drug_embedding': torch.tensor([128]),      # Drug representation
    'target_embedding': torch.tensor([128]),    # Target representation
    'pathway_embedding': torch.tensor([128]),   # Pathway representation
    'hierarchical_encoding': torch.tensor([128]) # Multi-level encoding
}
```

### 3. Perturbational Features
```python
# Example perturbational feature structure
perturbational_features = {
    'gene_expression': torch.tensor([978]),     # L1000 landmark genes
    'pathway_scores': torch.tensor([50]),       # Pathway activity
    'meta_signature': torch.tensor([978]),      # Consensus signature
    'perturbation_strength': torch.scalar      # Overall magnitude
}
```

### 4. Feature Quality Metrics
```
📈 Feature Quality Assessment:
├── Chemical graphs: 99.8% valid molecular graphs
├── MechTokens: 95% coverage of compound-target pairs
├── Perturbational: 85% compounds with LINCS data
├── Protein pockets: 75% targets with AlphaFold structures
└── Overall completeness: 92% multi-modal coverage
```

## 🛠️ Advanced Features

### 1. Counterfactual Fragment Analysis
```python
# Novel contribution: Identify causal molecular fragments
from moa.features.chemical import SubstructureCounterfactualAnalyzer

analyzer = SubstructureCounterfactualAnalyzer()
fragment_importance = analyzer.analyze_substructure_importance(
    smiles_list, moa_labels, moa_names
)
```

### 2. Hierarchical MechToken Encoding
```python
# Multi-level biological organization encoding
from moa.features.mechanism_tokens import HierarchicalEncoder

encoder = HierarchicalEncoder()
hierarchical_tokens = encoder.encode_biological_hierarchy(
    drug_targets, target_pathways, pathway_functions
)
```

### 3. Meta-Signature Generation
```python
# Consensus perturbational signatures across cell lines
from moa.features.perturbational import MetaSignatureGenerator

generator = MetaSignatureGenerator()
meta_signatures = generator.generate_consensus_signatures(
    lincs_signatures, cell_line_metadata
)
```

## 🧪 Validation & Quality Control

### Feature Validation Tests
```bash
# Run feature validation
python tests/test_feature_extraction.py

# Validate specific feature types
python -m pytest tests/test_chemical_features.py -v
python -m pytest tests/test_mechanism_tokens.py -v
python -m pytest tests/test_perturbational_features.py -v
```

### Quality Metrics
```python
# Check feature quality
from moa.features.feature_extractor import FeatureQualityAnalyzer

analyzer = FeatureQualityAnalyzer()
quality_report = analyzer.analyze_feature_quality(
    chemical_features, mechanism_tokens, perturbational_features
)
```

## 🔧 Troubleshooting

### Common Issues

1. **Memory Issues with Large Molecules**
   ```bash
   # Reduce batch size for large molecules
   python examples/phase2_feature_demo.py --batch_size 32
   ```

2. **Missing LINCS Data**
   ```bash
   # Use alternative perturbational databases
   configs:
     features:
       perturbational:
         fallback_databases: ["CMAP", "GDSC"]
   ```

3. **AlphaFold Structure Download**
   ```bash
   # Manual structure download
   python moa/features/download_alphafold.py --target_list targets.txt
   ```

### Performance Optimization
```bash
# Use GPU acceleration
export CUDA_VISIBLE_DEVICES=0
python examples/phase2_feature_demo.py --use_gpu

# Parallel processing
python examples/phase2_feature_demo.py --n_workers 8
```

## 📊 Feature Statistics

### Dimensionality Summary
```
Feature Type              | Dimensions | Coverage | Quality
--------------------------|------------|----------|--------
Chemical Graphs           | Variable   | 99.8%    | High
  - Node features         | 64/atom    | 100%     | High
  - Edge features         | 16/bond    | 100%     | High
  - Counterfactual weights| 1/atom     | 100%     | Novel
Mechanism Tokens          | 128        | 95%      | High
  - Drug embeddings       | 128        | 98%      | High
  - Target embeddings     | 128        | 92%      | High
  - Pathway embeddings    | 128        | 90%      | High
Perturbational Features   | 1,028      | 85%      | Medium
  - Gene expression       | 978        | 85%      | Medium
  - Pathway scores        | 50         | 90%      | High
Protein Pockets          | 256        | 75%      | Medium
  - Geometric features    | 128        | 80%      | High
  - Physicochemical       | 128        | 70%      | Medium
```

## 🔄 Next Steps

After completing Phase 2:

1. **Verify Feature Quality**: Check feature statistics and visualizations
2. **Review Feature Coverage**: Ensure adequate multi-modal coverage
3. **Proceed to Phase 3**: Model development with extracted features
4. **Optional**: Fine-tune feature extraction parameters

```bash
# Ready for Phase 3?
python examples/phase3_model_demo.py
```

## 📖 Related Documentation

- **Feature Format Specification**: `docs/feature_formats.md`
- **Chemical Feature Guide**: `docs/chemical_features.md`
- **MechToken Documentation**: `docs/mechanism_tokens.md`
- **API Reference**: `docs/api/features.html`

---

**Phase 2 Complete! ✅ Advanced multi-modal features ready for model development in Phase 3.**
