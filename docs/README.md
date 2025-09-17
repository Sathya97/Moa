# MoA Prediction Framework Documentation

## Overview

This repository contains a comprehensive **Mechanism of Action (MoA) Prediction Framework** using multi-modal deep learning. The framework combines chemical, biological, and structural information to predict drug mechanisms of action with state-of-the-art accuracy and interpretability.

## 🚀 Quick Start

### Prerequisites

```bash
# Python 3.8+ required
pip install -r requirements.txt

# For GPU support (recommended)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Basic Usage

```bash
# Run complete demo
python examples/complete_demo.py

# Run specific phase demos
python examples/phase1_foundations_demo.py
python examples/phase2_feature_demo.py
python examples/phase3_model_demo.py
python examples/phase4_training_demo.py
python examples/phase5_interpretation_demo.py
```

## 📁 Project Structure

```
Moa/
├── moa/                          # Core framework
│   ├── data/                     # Data processing modules
│   ├── features/                 # Feature extraction
│   ├── models/                   # Deep learning models
│   ├── training/                 # Training pipeline
│   ├── evaluation/               # Evaluation metrics
│   ├── interpretation/           # Model explainability
│   ├── applications/             # Practical applications
│   └── utils/                    # Utilities
├── examples/                     # Demonstration scripts
├── notebooks/                    # Jupyter notebooks
├── tests/                        # Test suites
├── configs/                      # Configuration files
├── docs/                         # Documentation
│   └── phase_guides/             # Phase-specific guides
└── outputs/                      # Generated outputs
```

## 🎯 Framework Phases

The framework is organized into 6 comprehensive phases:

| Phase | Description | Status | Guide |
|-------|-------------|--------|-------|
| **Phase 1** | Foundations & Data Collection | ✅ Complete | [Phase 1 Guide](docs/phase_guides/phase1_foundations.md) |
| **Phase 2** | Feature Engineering | ✅ Complete | [Phase 2 Guide](docs/phase_guides/phase2_features.md) |
| **Phase 3** | Model Development | ✅ Complete | [Phase 3 Guide](docs/phase_guides/phase3_models.md) |
| **Phase 4** | Training & Evaluation | ✅ Complete | [Phase 4 Guide](docs/phase_guides/phase4_training.md) |
| **Phase 5** | Interpretation & Applications | ✅ Complete | [Phase 5 Guide](docs/phase_guides/phase5_interpretation.md) |
| **Phase 6** | Publication & Deployment | 🔄 In Progress | [Phase 6 Guide](docs/phase_guides/phase6_deployment.md) |

## 🔬 Key Features

### Novel Methodological Contributions

1. **Substructure Counterfactual Analysis**: Identifies causal molecular fragments using counterfactual reasoning
2. **Mechanism Tokens (MechTokens)**: Ontology-aware embeddings encoding drug-target-pathway relationships
3. **Hypergraph Neural Networks**: Multi-modal fusion through drug-target-pathway-MoA hypergraphs
4. **Multi-Objective Learning**: Combines classification, prototype, invariance, and contrastive losses
5. **Comprehensive Interpretation**: Attention visualization, uncertainty estimation, and feature importance

### Practical Applications

- **Drug Repurposing**: Automated identification of new therapeutic uses
- **Knowledge Discovery**: Novel drug-pathway association discovery
- **Therapeutic Insights**: Target identification and combination prediction
- **Clinical Decision Support**: Evidence-based therapeutic recommendations
- **Biomarker Discovery**: Predictive biomarker identification

## 📊 Expected Results

### Model Performance
- **Multi-label AUROC**: 0.85-0.92 across MoA classes
- **Top-k Accuracy**: >90% for top-3 predictions
- **Calibration**: Well-calibrated uncertainty estimates
- **Interpretability**: High-quality attention and feature importance

### Applications
- **Repurposing Candidates**: 10-50 ranked candidates per query
- **Novel Associations**: 100+ statistically significant discoveries
- **Therapeutic Targets**: 5-15 promising targets per disease
- **Drug Combinations**: 20+ synergistic combinations identified

## 🛠️ Installation & Setup

### 1. Clone Repository
```bash
git clone <repository-url>
cd Moa
```

### 2. Install Dependencies
```bash
# Create virtual environment
python -m venv moa_env
source moa_env/bin/activate  # On Windows: moa_env\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### 3. Download Data (Optional)
```bash
# Download demo datasets
python moa/data/download_demo_data.py

# Or use your own data following the format in configs/data_config.yaml
```

### 4. Configure Settings
```bash
# Edit configuration
cp configs/config.yaml configs/my_config.yaml
# Modify configs/my_config.yaml as needed
```

## 🚀 Running the Framework

### Complete Workflow
```bash
# Run all phases sequentially
python examples/complete_workflow.py --config configs/my_config.yaml
```

### Individual Phases
```bash
# Phase 1: Data setup and preprocessing
python examples/phase1_foundations_demo.py

# Phase 2: Feature extraction
python examples/phase2_feature_demo.py

# Phase 3: Model training
python examples/phase3_model_demo.py

# Phase 4: Evaluation and comparison
python examples/phase4_training_demo.py

# Phase 5: Interpretation and applications
python examples/phase5_interpretation_demo.py
```

### Interactive Notebooks
```bash
# Launch Jupyter
jupyter notebook

# Open phase-specific notebooks
# notebooks/01_data_exploration.ipynb
# notebooks/02_feature_engineering.ipynb
# notebooks/03_model_architecture.ipynb
# notebooks/04_training_evaluation.ipynb
# notebooks/05_interpretation_applications.ipynb
```

## 📈 Monitoring & Outputs

### Generated Outputs
- **Models**: Trained model checkpoints in `outputs/models/`
- **Results**: Evaluation metrics in `outputs/results/`
- **Visualizations**: Plots and figures in `outputs/figures/`
- **Reports**: Comprehensive reports in `outputs/reports/`
- **Networks**: Repurposing networks in `outputs/networks/`

### Monitoring
- **TensorBoard**: Real-time training monitoring
- **Logs**: Detailed execution logs in `outputs/logs/`
- **Metrics**: Performance tracking in `outputs/metrics/`

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run specific test suites
pytest tests/test_features.py
pytest tests/test_models.py
pytest tests/test_training.py
pytest tests/test_interpretation.py
```

## 📚 Documentation

- **API Documentation**: Auto-generated docs in `docs/api/`
- **Phase Guides**: Detailed guides in `docs/phase_guides/`
- **Tutorials**: Step-by-step tutorials in `docs/tutorials/`
- **Examples**: Working examples in `examples/`

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Support

- **Issues**: Report bugs and request features via GitHub Issues
- **Discussions**: Join community discussions
- **Documentation**: Comprehensive guides in `docs/`

## 🏆 Citation

If you use this framework in your research, please cite:

```bibtex
@article{moa_framework_2024,
  title={Multi-Modal Deep Learning for Mechanism of Action Prediction},
  author={[Authors]},
  journal={[Journal]},
  year={2024}
}
```

## 🔗 Related Work

- **ChEMBL Database**: Chemical bioactivity data
- **LINCS L1000**: Gene expression perturbation data
- **PyTorch Geometric**: Graph neural network framework
- **RDKit**: Chemical informatics toolkit

---

**Ready to explore drug mechanisms with state-of-the-art AI? Start with Phase 1!** 🚀
