#!/usr/bin/env python3
"""
Data processing script for MoA prediction framework.

This script processes raw data downloaded from various sources, performs
SMILES standardization, duplicate removal, label processing, and creates
train/validation/test splits.
"""

import argparse
import logging
from pathlib import Path

from moa.data.processors import DataProcessor
from moa.data.validators import DataValidator
from moa.utils.config import Config, setup_paths
from moa.utils.logger import setup_logging


def main():
    """Main function for data processing script."""
    parser = argparse.ArgumentParser(description="Process data for MoA prediction")
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                       help="Path to configuration file")
    parser.add_argument("--input-dir", type=str, default="data/raw",
                       help="Input directory with raw data")
    parser.add_argument("--output-dir", type=str, default="data/processed",
                       help="Output directory for processed data")
    parser.add_argument("--create-splits", action="store_true",
                       help="Create train/val/test splits")
    parser.add_argument("--validate", action="store_true",
                       help="Validate processed data")
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
    
    # Create directories
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Processing data from {input_dir} to {output_dir}")
    
    # Initialize processor
    processor = DataProcessor(config)
    
    try:
        # Process all data
        processed_data = processor.process_all(input_dir, output_dir)
        
        if not processed_data:
            logger.error("No data was processed. Check input directory and data files.")
            return 1
        
        # Create splits if requested
        if args.create_splits:
            logger.info("Creating data splits...")
            splits_dir = output_dir / "splits"
            processor.create_splits(processed_data, splits_dir)
        
        # Validate data if requested
        if args.validate:
            logger.info("Validating processed data...")
            validator = DataValidator(config)
            validation_results = validator.validate_all(output_dir)
            
            # Print validation results
            all_passed = True
            for dataset, results in validation_results.items():
                logger.info(f"\n{dataset} validation results:")
                for check, passed in results.items():
                    status = "PASS" if passed else "FAIL"
                    logger.info(f"  {check}: {status}")
                    if not passed:
                        all_passed = False
            
            if all_passed:
                logger.info("All validation checks passed!")
            else:
                logger.warning("Some validation checks failed. Please review the data.")
                return 1
        
        logger.info("Data processing completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"Data processing failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
