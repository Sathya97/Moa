# Phase 1: Foundations & Data Collection

## Overview

Phase 1 establishes the foundation for the MoA prediction framework by setting up the project structure, defining the scope, and implementing data collection and curation pipelines.

## 🎯 Objectives

- Set up comprehensive project structure
- Define MoA prediction scope and configuration
- Implement data collection from ChEMBL and other sources
- Build robust data curation and preprocessing pipeline

## 📁 Components

### 1. Project Structure Setup
- **File**: `moa/utils/setup.py`
- **Purpose**: Creates standardized directory structure
- **Output**: Organized project hierarchy

### 2. Configuration Management
- **File**: `configs/config.yaml`
- **Purpose**: Centralized configuration for all framework components
- **Features**: Model parameters, data paths, training settings

### 3. Data Collection Scripts
- **File**: `moa/data/chembl_collector.py`
- **Purpose**: Downloads and processes ChEMBL mechanism data
- **Output**: Raw mechanism-compound mappings

### 4. Data Curation Pipeline
- **File**: `moa/data/curator.py`
- **Purpose**: Standardizes, cleans, and validates data
- **Output**: Curated datasets ready for feature extraction

## 🚀 Execution Instructions

### Prerequisites
```bash
# Install required packages
pip install rdkit-pypi chembl_webresource_client pandas numpy

# Ensure internet connection for data download
```

### Step 1: Run Foundation Demo
```bash
# Execute complete Phase 1 workflow
python examples/phase1_foundations_demo.py

# Expected runtime: 5-10 minutes
# Memory usage: ~2GB
```

### Step 2: Interactive Exploration
```bash
# Launch Jupyter notebook
jupyter notebook notebooks/01_data_exploration.ipynb

# Explore data collection and curation interactively
```

### Step 3: Custom Configuration
```bash
# Copy and modify configuration
cp configs/config.yaml configs/my_config.yaml

# Edit data paths and parameters as needed
vim configs/my_config.yaml
```

## 📊 Expected Results

### 1. Project Structure
```
Moa/
├── moa/
│   ├── data/           ✅ Created
│   ├── features/       ✅ Created
│   ├── models/         ✅ Created
│   ├── training/       ✅ Created
│   ├── evaluation/     ✅ Created
│   ├── interpretation/ ✅ Created
│   ├── applications/   ✅ Created
│   └── utils/          ✅ Created
├── configs/            ✅ Created
├── examples/           ✅ Created
├── notebooks/          ✅ Created
├── tests/              ✅ Created
└── outputs/            ✅ Created
```

### 2. Configuration Files
- **`configs/config.yaml`**: Main configuration (✅ Created)
- **`configs/data_config.yaml`**: Data-specific settings (✅ Created)
- **`requirements.txt`**: Python dependencies (✅ Created)

### 3. Data Collection Results
```
📊 ChEMBL Data Collection Summary:
├── Mechanisms collected: ~2,500 unique MoAs
├── Compounds processed: ~50,000 bioactive compounds
├── Target mappings: ~15,000 compound-target pairs
└── Pathway annotations: ~500 biological pathways
```

### 4. Data Curation Metrics
```
🔧 Data Curation Results:
├── SMILES standardization: 98% success rate
├── Duplicate removal: ~15% duplicates identified
├── Scaffold splits: Train/Val/Test (70/15/15)
└── Label balancing: Balanced across MoA classes
```

### 5. Generated Files
```
outputs/
├── data/
│   ├── raw_chembl_mechanisms.csv      # Raw ChEMBL data
│   ├── curated_compounds.csv          # Processed compounds
│   ├── moa_labels.csv                 # MoA annotations
│   └── train_val_test_splits.pkl      # Data splits
├── logs/
│   └── phase1_execution.log           # Detailed logs
└── reports/
    └── phase1_summary_report.txt      # Summary report
```

## 🔍 Key Outputs Explained

### 1. Curated Compounds Dataset
- **Format**: CSV with standardized SMILES, molecular properties
- **Size**: ~40,000 compounds after curation
- **Columns**: `compound_id`, `smiles`, `molecular_weight`, `logp`, `moa_labels`

### 2. MoA Label Matrix
- **Format**: Multi-label binary matrix
- **Dimensions**: [n_compounds, n_moa_classes]
- **Sparsity**: ~15% (realistic for biological data)

### 3. Data Splits
- **Training**: 70% of data (~28,000 compounds)
- **Validation**: 15% of data (~6,000 compounds)
- **Test**: 15% of data (~6,000 compounds)
- **Strategy**: Scaffold-based splitting for realistic evaluation

### 4. Quality Metrics
```
📈 Data Quality Assessment:
├── SMILES validity: 99.2%
├── Molecular weight range: 150-800 Da
├── Drug-like properties: 85% Lipinski compliant
├── MoA coverage: All major therapeutic classes
└── Label distribution: Balanced across classes
```

## 🛠️ Troubleshooting

### Common Issues

1. **ChEMBL Download Timeout**
   ```bash
   # Increase timeout in config
   chembl:
     timeout: 300  # Increase from default 60
   ```

2. **Memory Issues with Large Datasets**
   ```bash
   # Process in chunks
   python examples/phase1_foundations_demo.py --chunk_size 1000
   ```

3. **RDKit Installation Issues**
   ```bash
   # Alternative installation
   conda install -c conda-forge rdkit
   ```

### Validation Steps

```bash
# Verify data integrity
python -c "
import pandas as pd
df = pd.read_csv('outputs/data/curated_compounds.csv')
print(f'Compounds: {len(df)}')
print(f'Valid SMILES: {df.smiles.notna().sum()}')
"

# Check MoA distribution
python moa/data/validate_data.py --data_path outputs/data/
```

## 📚 Technical Details

### Data Sources
- **ChEMBL**: Primary source for compound-MoA mappings
- **PubChem**: Additional compound properties
- **UniProt**: Protein target information
- **Reactome**: Pathway annotations

### Processing Pipeline
1. **Download**: Fetch data from public databases
2. **Standardize**: Normalize SMILES and molecular representations
3. **Filter**: Remove invalid/problematic compounds
4. **Annotate**: Add MoA labels and metadata
5. **Split**: Create train/validation/test sets
6. **Validate**: Quality checks and statistics

### Performance Metrics
- **Processing Speed**: ~1,000 compounds/minute
- **Memory Usage**: ~100MB per 10,000 compounds
- **Storage**: ~500MB for complete curated dataset

## 🔄 Next Steps

After completing Phase 1:

1. **Verify Outputs**: Check all generated files in `outputs/data/`
2. **Review Logs**: Examine `outputs/logs/phase1_execution.log`
3. **Proceed to Phase 2**: Feature engineering with curated data
4. **Optional**: Customize configuration for your specific use case

```bash
# Ready for Phase 2?
python examples/phase2_feature_demo.py
```

## 📖 Related Documentation

- **Data Format Specification**: `docs/data_formats.md`
- **Configuration Guide**: `docs/configuration.md`
- **API Reference**: `docs/api/data.html`

---

**Phase 1 Complete! ✅ Ready for advanced feature engineering in Phase 2.**
