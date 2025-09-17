"""Data validation utilities."""

import logging
from pathlib import Path
from typing import Dict, List

import pandas as pd
from rdkit import Chem

from moa.utils.config import Config
from moa.utils.logger import get_logger

logger = get_logger(__name__)


class DataValidator:
    """Validator for data quality and consistency checks."""
    
    def __init__(self, config: Config):
        """
        Initialize data validator.
        
        Args:
            config: Configuration object
        """
        self.config = config
    
    def validate_smiles(self, df: pd.DataFrame, smiles_col: str = "standardized_smiles") -> Dict[str, bool]:
        """
        Validate SMILES in a DataFrame.
        
        Args:
            df: DataFrame to validate
            smiles_col: Name of SMILES column
            
        Returns:
            Dictionary of validation results
        """
        results = {}
        
        # Check if SMILES column exists
        results["smiles_column_exists"] = smiles_col in df.columns
        if not results["smiles_column_exists"]:
            return results
        
        # Check for missing SMILES
        missing_smiles = df[smiles_col].isna().sum()
        results["no_missing_smiles"] = missing_smiles == 0
        
        # Check SMILES validity
        valid_smiles = df[smiles_col].apply(self._is_valid_smiles).sum()
        results["all_smiles_valid"] = valid_smiles == len(df)
        
        # Check for duplicates
        duplicates = df[smiles_col].duplicated().sum()
        results["no_duplicate_smiles"] = duplicates == 0
        
        logger.info(f"SMILES validation: {missing_smiles} missing, "
                   f"{len(df) - valid_smiles} invalid, {duplicates} duplicates")
        
        return results
    
    def validate_labels(self, df: pd.DataFrame, label_cols: List[str] = None) -> Dict[str, bool]:
        """
        Validate MoA labels in a DataFrame.
        
        Args:
            df: DataFrame to validate
            label_cols: List of label columns to check
            
        Returns:
            Dictionary of validation results
        """
        results = {}
        
        if label_cols is None:
            # Find MoA label columns
            label_cols = [col for col in df.columns if col.startswith("moa_")]
        
        # Check if label columns exist
        results["label_columns_exist"] = len(label_cols) > 0
        if not results["label_columns_exist"]:
            return results
        
        # Check for missing labels
        missing_labels = df[label_cols].isna().all(axis=1).sum()
        results["no_missing_labels"] = missing_labels == 0
        
        # Check label distribution
        label_counts = df[label_cols].sum()
        min_samples_per_label = self.config.get("validation.min_samples_per_label", 10)
        # Defensive programming: ensure min_samples_per_label is not None
        if min_samples_per_label is None:
            min_samples_per_label = 10
        results["sufficient_samples_per_label"] = (label_counts >= min_samples_per_label).all()
        
        # Check for empty labels (all zeros)
        empty_labels = (df[label_cols].sum(axis=1) == 0).sum()
        results["no_empty_labels"] = empty_labels == 0
        
        logger.info(f"Label validation: {missing_labels} missing, "
                   f"{empty_labels} empty, min samples: {label_counts.min()}")
        
        return results
    
    def validate_dataset_size(self, df: pd.DataFrame) -> Dict[str, bool]:
        """
        Validate dataset size requirements.

        Args:
            df: DataFrame to validate

        Returns:
            Dictionary of validation results
        """
        results = {}

        min_samples = self.config.get("validation.min_dataset_size", 1000)
        # Defensive programming: ensure min_samples is not None
        if min_samples is None:
            min_samples = 1000

        results["sufficient_dataset_size"] = len(df) >= min_samples

        logger.info(f"Dataset size validation: {len(df)} samples (min: {min_samples})")

        return results
    
    def validate_data_consistency(self, df: pd.DataFrame) -> Dict[str, bool]:
        """
        Validate data consistency.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Dictionary of validation results
        """
        results = {}
        
        # Check for required columns
        required_cols = ["molecule_chembl_id", "standardized_smiles"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        results["required_columns_present"] = len(missing_cols) == 0
        
        # Check for data type consistency
        if "molecule_chembl_id" in df.columns:
            results["chembl_id_format_valid"] = df["molecule_chembl_id"].str.startswith("CHEMBL").all()
        
        logger.info(f"Consistency validation: missing columns: {missing_cols}")
        
        return results
    
    def validate_splits(self, splits_dir: Path) -> Dict[str, bool]:
        """
        Validate train/val/test splits.
        
        Args:
            splits_dir: Directory containing split files
            
        Returns:
            Dictionary of validation results
        """
        results = {}
        
        # Check if split files exist
        split_files = ["train.csv", "val.csv", "test.csv"]
        existing_files = [f for f in split_files if (splits_dir / f).exists()]
        results["all_split_files_exist"] = len(existing_files) == len(split_files)
        
        if not results["all_split_files_exist"]:
            logger.warning(f"Missing split files: {set(split_files) - set(existing_files)}")
            return results
        
        # Load splits and check sizes
        train_df = pd.read_csv(splits_dir / "train.csv")
        val_df = pd.read_csv(splits_dir / "val.csv")
        test_df = pd.read_csv(splits_dir / "test.csv")
        
        total_samples = len(train_df) + len(val_df) + len(test_df)
        train_ratio = len(train_df) / total_samples
        val_ratio = len(val_df) / total_samples
        test_ratio = len(test_df) / total_samples
        
        # Check split ratios
        expected_train = self.config.get("evaluation.splits.scaffold_split.train_ratio", 0.7)
        expected_val = self.config.get("evaluation.splits.scaffold_split.val_ratio", 0.15)
        expected_test = self.config.get("evaluation.splits.scaffold_split.test_ratio", 0.15)

        # Defensive programming: ensure values are not None
        if expected_train is None:
            expected_train = 0.7
        if expected_val is None:
            expected_val = 0.15
        if expected_test is None:
            expected_test = 0.15
        
        ratio_tolerance = 0.05
        results["train_ratio_correct"] = abs(train_ratio - expected_train) < ratio_tolerance
        results["val_ratio_correct"] = abs(val_ratio - expected_val) < ratio_tolerance
        results["test_ratio_correct"] = abs(test_ratio - expected_test) < ratio_tolerance
        
        # Check for data leakage (no overlapping SMILES)
        train_smiles = set(train_df["standardized_smiles"])
        val_smiles = set(val_df["standardized_smiles"])
        test_smiles = set(test_df["standardized_smiles"])
        
        results["no_train_val_overlap"] = len(train_smiles & val_smiles) == 0
        results["no_train_test_overlap"] = len(train_smiles & test_smiles) == 0
        results["no_val_test_overlap"] = len(val_smiles & test_smiles) == 0
        
        logger.info(f"Split validation: train={train_ratio:.3f}, val={val_ratio:.3f}, test={test_ratio:.3f}")
        
        return results
    
    def validate_all(self, data_dir: Path) -> Dict[str, Dict[str, bool]]:
        """
        Run all validation checks.
        
        Args:
            data_dir: Directory containing data to validate
            
        Returns:
            Dictionary of validation results by dataset
        """
        results = {}
        
        # Validate processed data
        processed_file = data_dir / "chembl_processed.csv"
        if processed_file.exists():
            df = pd.read_csv(processed_file)
            
            dataset_results = {}
            dataset_results.update(self.validate_smiles(df))
            dataset_results.update(self.validate_labels(df))
            dataset_results.update(self.validate_dataset_size(df))
            dataset_results.update(self.validate_data_consistency(df))
            
            results["chembl_processed"] = dataset_results
        
        # Validate splits
        splits_dir = data_dir / "splits"
        if splits_dir.exists():
            results["splits"] = self.validate_splits(splits_dir)
        
        return results
    
    def _is_valid_smiles(self, smiles: str) -> bool:
        """Check if a SMILES string is valid."""
        try:
            mol = Chem.MolFromSmiles(smiles)
            return mol is not None
        except Exception:
            return False
