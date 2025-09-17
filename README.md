# Mechanism of Action (MoA) Prediction Framework

A comprehensive framework for predicting drug mechanisms of action using multi-modal deep learning approaches that combine chemical, biological, and structural information.

## Overview

This project implements a novel approach to MoA prediction that leverages:

- **Chemical Features**: Graph neural networks with substructure counterfactual analysis
- **Mechanism Tokens (MechTokens)**: Ontology-aware embeddings of drug-target-pathway relationships
- **Perturbational Biology**: LINCS L1000 gene expression signatures and pathway activity scores
- **Protein Structure**: Optional 3D binding site features from AlphaFold/PDBe
- **Hypergraph Architecture**: Multi-modal fusion through drug-target-pathway-MoA hypergraphs

## Key Features

### Novel Contributions
- **Substructure Counterfactual Scoring**: Identifies causal molecular fragments for MoA prediction
- **MechTokens**: Ontology-aware embeddings that encode drug-target-pathway relationships
- **Hypergraph Fusion**: Advanced multi-modal integration using hypergraph attention
- **Multi-objective Learning**: Combines classification, prototype, invariance, and contrastive losses

### Supported Modalities
- ✅ Chemical structure (SMILES, molecular graphs)
- ✅ Gene expression perturbations (LINCS L1000)
- ✅ Protein targets and pathways (ChEMBL, Reactome, KEGG)
- ⚠️ 3D protein structures (optional, experimental)

### Benchmarks
- ChEMBL mechanism annotations
- DrugBank mechanism classifications
- LINCS L1000 perturbation profiles

## Project Structure

```
moa/
├── moa/                    # Main package
│   ├── data/              # Data loading and processing
│   ├── features/          # Feature engineering modules
│   ├── models/            # Model architectures
│   ├── training/          # Training and optimization
│   ├── evaluation/        # Evaluation metrics and analysis
│   ├── utils/             # Utilities and configuration
│   └── cli/               # Command-line interfaces
├── configs/               # Configuration files
├── data/                  # Data storage
│   ├── raw/              # Raw downloaded data
│   ├── processed/        # Processed datasets
│   └── splits/           # Train/val/test splits
├── experiments/          # Experiment results and logs
├── notebooks/            # Jupyter notebooks for analysis
├── scripts/              # Data collection and processing scripts
└── tests/                # Unit tests

```

## Installation

### Prerequisites
- Python 3.8+
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)

### Setup
```bash
# Clone the repository
git clone https://github.com/your-org/moa-prediction.git
cd moa-prediction

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

## Quick Start

### 1. Configuration
The framework uses YAML configuration files. See `configs/config.yaml` for all available options:

```yaml
# Enable/disable modalities
modalities:
  chemistry: true
  perturbation: true
  targets: true
  pathways: true
  structures: false

# Model architecture
model:
  architecture: "hypergraph_transformer"
  chemical_branch:
    type: "graph_transformer"
    hidden_dim: 512
```

### 2. Data Collection
```bash
# Download and process ChEMBL data
moa-data collect --source chembl --version 33

# Download LINCS L1000 data
moa-data collect --source lincs --level L3

# Process and create splits
moa-data process --create-splits
```

### 3. Training
```bash
# Train with default configuration
moa-train --config configs/config.yaml

# Train with custom settings
moa-train --config configs/config.yaml \
          --model.hidden_dim 256 \
          --training.batch_size 64
```

### 4. Prediction
```bash
# Predict MoA for new compounds
moa-predict --model models/best_model.ckpt \
            --input compounds.smi \
            --output predictions.csv
```

## Research Roadmap

### Phase 1: Foundations ✅
- [x] Project structure and configuration
- [ ] Data collection scripts (ChEMBL, LINCS, pathways)
- [ ] Data curation pipeline (SMILES standardization, splits)

### Phase 2: Feature Engineering
- [ ] Chemical graph features with counterfactual analysis
- [ ] MechTokens: ontology-aware embeddings
- [ ] LINCS meta-signatures and pathway scoring
- [ ] Protein pocket features (optional)

### Phase 3: Model Development
- [ ] Graph Transformer with substructure gates
- [ ] Pathway Transformer for biological features
- [ ] Hypergraph fusion layer
- [ ] Multi-objective loss functions

### Phase 4: Training & Evaluation
- [ ] Curriculum learning implementation
- [ ] Comprehensive evaluation metrics
- [ ] Baseline model comparisons
- [ ] Out-of-distribution testing

### Phase 5: Interpretation & Applications
- [ ] Explainability tools (substructure attribution)
- [ ] Uncertainty estimation
- [ ] Drug repurposing pipeline
- [ ] Knowledge discovery applications

### Phase 6: Publication & Deployment
- [ ] Reproducibility package
- [ ] Web API and interface
- [ ] Documentation and tutorials

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this framework in your research, please cite:

```bibtex
@article{moa_prediction_2024,
  title={Multi-modal Mechanism of Action Prediction using Hypergraph Neural Networks},
  author={MoA Research Team},
  journal={Journal of Chemical Information and Modeling},
  year={2024},
  note={In preparation}
}
```

## Contact

- Research Team: research@moa-prediction.org
- Issues: [GitHub Issues](https://github.com/your-org/moa-prediction/issues)
- Discussions: [GitHub Discussions](https://github.com/your-org/moa-prediction/discussions)