#!/usr/bin/env python3
"""
Data download script for MoA prediction framework.

This script downloads data from various sources including ChEMBL, LINCS, 
Reactome, and other databases specified in the configuration.
"""

import argparse
import logging
from pathlib import Path
from typing import List, Optional

import pandas as pd

from moa.data.collectors import DataCollectorFactory
from moa.utils.config import Config, get_project_root, setup_paths
from moa.utils.logger import setup_logging


def download_chembl_data(config: Config, output_dir: Path, data_types: Optional[List[str]] = None) -> None:
    """
    Download ChEMBL data.
    
    Args:
        config: Configuration object
        output_dir: Output directory for data
        data_types: List of data types to download
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting ChEMBL data download...")
    
    collector = DataCollectorFactory.create_collector('chembl', config, output_dir / 'cache')
    
    if data_types is None:
        data_types = ['mechanisms', 'activities', 'targets', 'compounds']
    
    results = collector.collect(data_types)
    
    # Save data to files
    chembl_dir = output_dir / 'chembl'
    chembl_dir.mkdir(parents=True, exist_ok=True)
    
    for data_type, df in results.items():
        output_file = chembl_dir / f'{data_type}.csv'
        df.to_csv(output_file, index=False)
        logger.info(f"Saved {len(df)} {data_type} records to {output_file}")
    
    logger.info("ChEMBL data download completed")


def download_lincs_data(config: Config, output_dir: Path) -> None:
    """
    Download LINCS data.
    
    Args:
        config: Configuration object
        output_dir: Output directory for data
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting LINCS data download...")
    
    collector = DataCollectorFactory.create_collector('lincs', config, output_dir / 'cache')
    
    signatures = collector.collect()
    
    # Save data to files
    lincs_dir = output_dir / 'lincs'
    lincs_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = lincs_dir / 'signatures.csv'
    signatures.to_csv(output_file, index=False)
    logger.info(f"Saved {len(signatures)} signature records to {output_file}")
    
    logger.info("LINCS data download completed")


def download_reactome_data(config: Config, output_dir: Path, uniprot_ids: Optional[List[str]] = None) -> None:
    """
    Download Reactome data.
    
    Args:
        config: Configuration object
        output_dir: Output directory for data
        uniprot_ids: Optional list of UniProt IDs for protein-pathway mapping
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting Reactome data download...")
    
    collector = DataCollectorFactory.create_collector('reactome', config, output_dir / 'cache')
    
    kwargs = {}
    if uniprot_ids:
        kwargs['uniprot_ids'] = uniprot_ids
    
    results = collector.collect(**kwargs)
    
    # Save data to files
    reactome_dir = output_dir / 'reactome'
    reactome_dir.mkdir(parents=True, exist_ok=True)
    
    for data_type, df in results.items():
        output_file = reactome_dir / f'{data_type}.csv'
        df.to_csv(output_file, index=False)
        logger.info(f"Saved {len(df)} {data_type} records to {output_file}")
    
    logger.info("Reactome data download completed")


def create_integrated_dataset(config: Config, data_dir: Path) -> None:
    """
    Create integrated dataset by linking data from different sources.
    
    Args:
        config: Configuration object
        data_dir: Data directory containing downloaded data
    """
    logger = logging.getLogger(__name__)
    logger.info("Creating integrated dataset...")
    
    # Load ChEMBL data
    chembl_dir = data_dir / 'chembl'
    
    try:
        mechanisms = pd.read_csv(chembl_dir / 'mechanisms.csv')
        activities = pd.read_csv(chembl_dir / 'activities.csv')
        targets = pd.read_csv(chembl_dir / 'targets.csv')
        compounds = pd.read_csv(chembl_dir / 'compounds.csv')
        
        logger.info(f"Loaded ChEMBL data: {len(mechanisms)} mechanisms, "
                   f"{len(activities)} activities, {len(targets)} targets, "
                   f"{len(compounds)} compounds")
        
        # Create compound-MoA mappings
        compound_moa = mechanisms.merge(
            compounds[['molecule_chembl_id', 'canonical_smiles']], 
            on='molecule_chembl_id', 
            how='inner'
        )
        
        # Add target information
        compound_moa = compound_moa.merge(
            targets[['target_chembl_id', 'pref_name', 'target_type']], 
            on='target_chembl_id', 
            how='left'
        )
        
        # Save integrated dataset
        output_file = data_dir / 'integrated_compound_moa.csv'
        compound_moa.to_csv(output_file, index=False)
        logger.info(f"Saved integrated dataset with {len(compound_moa)} records to {output_file}")
        
    except FileNotFoundError as e:
        logger.error(f"Required data file not found: {e}")
        logger.error("Please run data download first")


def main():
    """Main function for data download script."""
    parser = argparse.ArgumentParser(description="Download data for MoA prediction")
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                       help="Path to configuration file")
    parser.add_argument("--sources", nargs="+", default=["chembl", "reactome"],
                       choices=["chembl", "lincs", "reactome"],
                       help="Data sources to download")
    parser.add_argument("--chembl-types", nargs="+", 
                       default=["mechanisms", "activities", "targets"],
                       choices=["mechanisms", "activities", "targets", "compounds"],
                       help="ChEMBL data types to download")
    parser.add_argument("--output-dir", type=str, default="data/raw",
                       help="Output directory for downloaded data")
    parser.add_argument("--integrate", action="store_true",
                       help="Create integrated dataset after download")
    parser.add_argument("--log-level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(level=getattr(logging, args.log_level))
    logger = logging.getLogger(__name__)
    
    # Load configuration
    config = Config(args.config)
    
    # Setup paths
    setup_paths(config)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Starting data download to {output_dir}")
    logger.info(f"Data sources: {args.sources}")
    
    # Download data from each source
    for source in args.sources:
        try:
            if source == "chembl":
                download_chembl_data(config, output_dir, args.chembl_types)
            elif source == "lincs":
                download_lincs_data(config, output_dir)
            elif source == "reactome":
                # Get UniProt IDs from ChEMBL targets if available
                uniprot_ids = None
                chembl_targets_file = output_dir / 'chembl' / 'targets.csv'
                if chembl_targets_file.exists():
                    targets_df = pd.read_csv(chembl_targets_file)
                    if 'target_components' in targets_df.columns:
                        # Extract UniProt IDs from target components
                        # This is simplified - actual implementation would parse the JSON
                        uniprot_ids = []
                
                download_reactome_data(config, output_dir, uniprot_ids)
                
        except Exception as e:
            logger.error(f"Failed to download {source} data: {e}")
            continue
    
    # Create integrated dataset if requested
    if args.integrate:
        try:
            create_integrated_dataset(config, output_dir)
        except Exception as e:
            logger.error(f"Failed to create integrated dataset: {e}")
    
    logger.info("Data download completed")


if __name__ == "__main__":
    main()
