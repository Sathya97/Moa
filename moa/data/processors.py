"""Data processing and curation utilities."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

from moa.utils.config import Config
from moa.utils.logger import get_logger

logger = get_logger(__name__)


class SMILESProcessor:
    """Processor for SMILES standardization and validation."""
    
    def __init__(self, config: Config):
        """
        Initialize SMILES processor.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.standardizer = config.get("processing.smiles.standardizer", "rdkit")
        self.remove_salts = config.get("processing.smiles.remove_salts", True)
        self.neutralize = config.get("processing.smiles.neutralize", True)
        self.canonicalize = config.get("processing.smiles.canonicalize", True)
    
    def standardize_smiles(self, smiles: str) -> Optional[str]:
        """
        Standardize a SMILES string.
        
        Args:
            smiles: Input SMILES string
            
        Returns:
            Standardized SMILES or None if invalid
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            
            # Remove salts
            if self.remove_salts:
                mol = Chem.rdMolStandardize.FragmentParent(mol)
            
            # Neutralize
            if self.neutralize:
                uncharger = Chem.rdMolStandardize.Uncharger()
                mol = uncharger.uncharge(mol)
            
            # Canonicalize
            if self.canonicalize:
                return Chem.MolToSmiles(mol, canonical=True)
            else:
                return Chem.MolToSmiles(mol)
                
        except Exception:
            return None
    
    def validate_smiles(self, smiles: str) -> bool:
        """
        Validate a SMILES string.
        
        Args:
            smiles: SMILES string to validate
            
        Returns:
            True if valid, False otherwise
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return False
            
            # Check molecular properties
            mw = Descriptors.MolWt(mol)
            heavy_atoms = mol.GetNumHeavyAtoms()
            
            max_mw = self.config.get("processing.quality.max_molecular_weight", 1000)
            min_mw = self.config.get("processing.quality.min_molecular_weight", 150)
            max_heavy = self.config.get("processing.quality.max_heavy_atoms", 100)
            min_heavy = self.config.get("processing.quality.min_heavy_atoms", 5)
            
            if not (min_mw <= mw <= max_mw):
                return False
            if not (min_heavy <= heavy_atoms <= max_heavy):
                return False
            
            return True
            
        except Exception:
            return False
    
    def process_smiles_column(self, df: pd.DataFrame, smiles_col: str = "canonical_smiles") -> pd.DataFrame:
        """
        Process SMILES column in a DataFrame.
        
        Args:
            df: Input DataFrame
            smiles_col: Name of SMILES column
            
        Returns:
            Processed DataFrame
        """
        logger.info(f"Processing {len(df)} SMILES...")
        
        # Standardize SMILES
        df["standardized_smiles"] = df[smiles_col].apply(self.standardize_smiles)
        
        # Remove invalid SMILES
        valid_mask = df["standardized_smiles"].notna()
        logger.info(f"Removed {(~valid_mask).sum()} invalid SMILES")
        
        df = df[valid_mask].copy()
        
        # Validate SMILES
        df["is_valid"] = df["standardized_smiles"].apply(self.validate_smiles)
        valid_mask = df["is_valid"]
        logger.info(f"Removed {(~valid_mask).sum()} SMILES failing quality filters")
        
        df = df[valid_mask].copy()
        df = df.drop(columns=["is_valid"])
        
        return df


class DuplicateHandler:
    """Handler for duplicate removal."""
    
    def __init__(self, config: Config):
        """
        Initialize duplicate handler.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.method = config.get("processing.duplicates.method", "inchikey")
        self.keep = config.get("processing.duplicates.keep", "first")
    
    def remove_duplicates(self, df: pd.DataFrame, smiles_col: str = "standardized_smiles") -> pd.DataFrame:
        """
        Remove duplicate compounds.
        
        Args:
            df: Input DataFrame
            smiles_col: Name of SMILES column
            
        Returns:
            DataFrame with duplicates removed
        """
        logger.info(f"Removing duplicates using {self.method} method...")
        
        if self.method == "inchikey":
            # Generate InChI keys
            df["inchi_key"] = df[smiles_col].apply(self._smiles_to_inchikey)
            duplicate_col = "inchi_key"
        elif self.method == "smiles":
            duplicate_col = smiles_col
        else:
            raise ValueError(f"Unknown duplicate method: {self.method}")
        
        # Remove duplicates
        initial_count = len(df)
        df = df.drop_duplicates(subset=[duplicate_col], keep=self.keep)
        removed_count = initial_count - len(df)
        
        logger.info(f"Removed {removed_count} duplicates")
        
        if self.method == "inchikey":
            df = df.drop(columns=["inchi_key"])
        
        return df
    
    def _smiles_to_inchikey(self, smiles: str) -> Optional[str]:
        """Convert SMILES to InChI key."""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            return Chem.MolToInchiKey(mol)
        except Exception:
            return None


class LabelProcessor:
    """Processor for MoA labels."""
    
    def __init__(self, config: Config):
        """
        Initialize label processor.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.prediction_type = config.get("scope.prediction_type", "multi_label")
    
    def process_moa_labels(self, df: pd.DataFrame, moa_col: str = "mechanism_of_action") -> pd.DataFrame:
        """
        Process mechanism of action labels.
        
        Args:
            df: Input DataFrame
            moa_col: Name of MoA column
            
        Returns:
            DataFrame with processed labels
        """
        logger.info("Processing MoA labels...")
        
        # Clean and standardize MoA descriptions
        df["moa_cleaned"] = df[moa_col].str.strip().str.lower()
        
        # Remove entries without MoA information
        valid_mask = df["moa_cleaned"].notna() & (df["moa_cleaned"] != "")
        logger.info(f"Removed {(~valid_mask).sum()} entries without MoA information")
        df = df[valid_mask].copy()
        
        if self.prediction_type == "multi_label":
            # Split multiple MoAs (assuming they're separated by semicolons or commas)
            df["moa_list"] = df["moa_cleaned"].str.split(r"[;,]")
            df["moa_list"] = df["moa_list"].apply(lambda x: [moa.strip() for moa in x if moa.strip()])
            
            # Create binary label matrix
            mlb = MultiLabelBinarizer()
            label_matrix = mlb.fit_transform(df["moa_list"])
            
            # Create label columns
            label_cols = [f"moa_{label}" for label in mlb.classes_]
            label_df = pd.DataFrame(label_matrix, columns=label_cols, index=df.index)
            
            df = pd.concat([df, label_df], axis=1)
            df["moa_classes"] = list(mlb.classes_)
            
        else:  # single_label
            # Use the first MoA if multiple are present
            df["moa_primary"] = df["moa_cleaned"].str.split(r"[;,]").str[0].str.strip()
        
        return df


class DataSplitter:
    """Data splitter for creating train/val/test splits."""
    
    def __init__(self, config: Config):
        """
        Initialize data splitter.
        
        Args:
            config: Configuration object
        """
        self.config = config
    
    def create_scaffold_split(self, df: pd.DataFrame, smiles_col: str = "standardized_smiles") -> Dict[str, pd.DataFrame]:
        """
        Create scaffold-based split.
        
        Args:
            df: Input DataFrame
            smiles_col: Name of SMILES column
            
        Returns:
            Dictionary with train/val/test DataFrames
        """
        logger.info("Creating scaffold-based split...")
        
        # Generate Murcko scaffolds
        df["scaffold"] = df[smiles_col].apply(self._get_murcko_scaffold)
        
        # Get unique scaffolds
        scaffolds = df["scaffold"].unique()
        scaffolds = scaffolds[pd.notna(scaffolds)]
        
        # Split scaffolds
        train_ratio = self.config.get("evaluation.splits.scaffold_split.train_ratio", 0.7)
        val_ratio = self.config.get("evaluation.splits.scaffold_split.val_ratio", 0.15)
        
        train_scaffolds, temp_scaffolds = train_test_split(
            scaffolds, train_size=train_ratio, random_state=42
        )
        
        val_scaffolds, test_scaffolds = train_test_split(
            temp_scaffolds, train_size=val_ratio/(1-train_ratio), random_state=42
        )
        
        # Create splits
        train_df = df[df["scaffold"].isin(train_scaffolds)].copy()
        val_df = df[df["scaffold"].isin(val_scaffolds)].copy()
        test_df = df[df["scaffold"].isin(test_scaffolds)].copy()
        
        logger.info(f"Scaffold split: {len(train_df)} train, {len(val_df)} val, {len(test_df)} test")
        
        return {
            "train": train_df.drop(columns=["scaffold"]),
            "val": val_df.drop(columns=["scaffold"]),
            "test": test_df.drop(columns=["scaffold"])
        }
    
    def _get_murcko_scaffold(self, smiles: str) -> Optional[str]:
        """Get Murcko scaffold for a SMILES."""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            scaffold = rdMolDescriptors.GetMorganFingerprint(mol, 2)
            return str(scaffold)
        except Exception:
            return None


class DataProcessor:
    """Main data processor that orchestrates all processing steps."""
    
    def __init__(self, config: Config):
        """
        Initialize data processor.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.smiles_processor = SMILESProcessor(config)
        self.duplicate_handler = DuplicateHandler(config)
        self.label_processor = LabelProcessor(config)
        self.data_splitter = DataSplitter(config)
    
    def process_chembl_data(self, input_dir: Path) -> pd.DataFrame:
        """
        Process ChEMBL data.
        
        Args:
            input_dir: Directory containing raw ChEMBL data
            
        Returns:
            Processed DataFrame
        """
        logger.info("Processing ChEMBL data...")
        
        # Load data
        mechanisms = pd.read_csv(input_dir / "chembl" / "mechanisms.csv")
        compounds = pd.read_csv(input_dir / "chembl" / "compounds.csv")
        
        # Merge mechanisms with compounds
        df = mechanisms.merge(
            compounds[["molecule_chembl_id", "canonical_smiles"]], 
            on="molecule_chembl_id", 
            how="inner"
        )
        
        logger.info(f"Loaded {len(df)} compound-mechanism pairs")
        
        # Process SMILES
        df = self.smiles_processor.process_smiles_column(df)
        
        # Remove duplicates
        df = self.duplicate_handler.remove_duplicates(df)
        
        # Process labels
        df = self.label_processor.process_moa_labels(df)
        
        return df
    
    def process_all(self, input_dir: Path, output_dir: Path) -> Dict[str, pd.DataFrame]:
        """
        Process all data sources.
        
        Args:
            input_dir: Directory containing raw data
            output_dir: Directory for processed data
            
        Returns:
            Dictionary of processed DataFrames
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = {}
        
        # Process ChEMBL data
        if (input_dir / "chembl").exists():
            chembl_data = self.process_chembl_data(input_dir)
            results["chembl"] = chembl_data
            
            # Save processed data
            output_file = output_dir / "chembl_processed.csv"
            chembl_data.to_csv(output_file, index=False)
            logger.info(f"Saved processed ChEMBL data to {output_file}")
        
        return results
    
    def create_splits(self, data: Dict[str, pd.DataFrame], output_dir: Path) -> None:
        """
        Create train/val/test splits.
        
        Args:
            data: Dictionary of processed DataFrames
            output_dir: Directory for split data
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for dataset_name, df in data.items():
            logger.info(f"Creating splits for {dataset_name}")
            
            # Create scaffold split
            splits = self.data_splitter.create_scaffold_split(df)
            
            # Save splits
            for split_name, split_df in splits.items():
                output_file = output_dir / f"{dataset_name}_{split_name}.csv"
                split_df.to_csv(output_file, index=False)
                logger.info(f"Saved {split_name} split to {output_file}")
