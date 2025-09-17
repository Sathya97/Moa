#!/usr/bin/env python3
"""
Command-line interface for data operations.
"""

import argparse
import logging
import sys
from pathlib import Path

from moa.utils.config import Config
from moa.utils.logger import setup_logging


def collect_data(args):
    """Collect data from external sources."""
    logger = logging.getLogger(__name__)
    logger.info(f"Collecting data from {args.source}")
    
    # Import here to avoid circular imports
    from moa.data.collectors import DataCollectorFactory
    
    config = Config(args.config)
    collector = DataCollectorFactory.create_collector(args.source, config)
    
    if args.source == "chembl":
        data_types = args.chembl_types or ["mechanisms", "activities", "targets"]
        results = collector.collect(data_types)
        
        # Save results
        output_dir = Path(args.output) / "chembl"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for data_type, df in results.items():
            output_file = output_dir / f"{data_type}.csv"
            df.to_csv(output_file, index=False)
            logger.info(f"Saved {len(df)} {data_type} to {output_file}")
    
    elif args.source == "lincs":
        signatures = collector.collect()
        
        output_dir = Path(args.output) / "lincs"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / "signatures.csv"
        signatures.to_csv(output_file, index=False)
        logger.info(f"Saved {len(signatures)} signatures to {output_file}")
    
    else:
        logger.error(f"Unsupported source: {args.source}")
        return 1
    
    return 0


def process_data(args):
    """Process and curate collected data."""
    logger = logging.getLogger(__name__)
    logger.info("Processing collected data")
    
    # Import here to avoid circular imports
    from moa.data.processors import DataProcessor
    
    config = Config(args.config)
    processor = DataProcessor(config)
    
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    
    # Process data
    processed_data = processor.process_all(input_dir, output_dir)
    
    if args.create_splits:
        logger.info("Creating data splits")
        processor.create_splits(processed_data, output_dir / "splits")
    
    return 0


def validate_data(args):
    """Validate data quality and consistency."""
    logger = logging.getLogger(__name__)
    logger.info("Validating data")
    
    # Import here to avoid circular imports
    from moa.data.validators import DataValidator
    
    config = Config(args.config)
    validator = DataValidator(config)
    
    data_dir = Path(args.data_dir)
    
    # Run validation
    validation_results = validator.validate_all(data_dir)
    
    # Print results
    for dataset, results in validation_results.items():
        logger.info(f"\n{dataset} validation results:")
        for check, passed in results.items():
            status = "PASS" if passed else "FAIL"
            logger.info(f"  {check}: {status}")
    
    # Return non-zero if any validation failed
    all_passed = all(
        all(results.values()) for results in validation_results.values()
    )
    return 0 if all_passed else 1


def main():
    """Main CLI entry point for data operations."""
    parser = argparse.ArgumentParser(description="MoA Data Operations CLI")
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                       help="Path to configuration file")
    parser.add_argument("--log-level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Collect command
    collect_parser = subparsers.add_parser("collect", help="Collect data from external sources")
    collect_parser.add_argument("--source", type=str, required=True,
                               choices=["chembl", "lincs", "reactome"],
                               help="Data source to collect from")
    collect_parser.add_argument("--output", type=str, default="data/raw",
                               help="Output directory")
    collect_parser.add_argument("--chembl-types", nargs="+",
                               choices=["mechanisms", "activities", "targets", "compounds"],
                               help="ChEMBL data types to collect")
    collect_parser.set_defaults(func=collect_data)
    
    # Process command
    process_parser = subparsers.add_parser("process", help="Process and curate data")
    process_parser.add_argument("--input", type=str, default="data/raw",
                               help="Input directory with raw data")
    process_parser.add_argument("--output", type=str, default="data/processed",
                               help="Output directory for processed data")
    process_parser.add_argument("--create-splits", action="store_true",
                               help="Create train/val/test splits")
    process_parser.set_defaults(func=process_data)
    
    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate data quality")
    validate_parser.add_argument("--data-dir", type=str, default="data/processed",
                               help="Directory containing data to validate")
    validate_parser.set_defaults(func=validate_data)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Setup logging
    setup_logging(level=getattr(logging, args.log_level))
    
    # Run command
    try:
        return args.func(args)
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Command failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
