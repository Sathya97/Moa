"""
Data loading utilities for MoA prediction training.

This module provides efficient data loading, batching, and preprocessing
for multi-modal MoA prediction models.
"""

from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from torch_geometric.data import Data, Batch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

from moa.utils.config import Config
from moa.utils.logger import get_logger

logger = get_logger(__name__)


class MoADataset(Dataset):
    """Dataset class for MoA prediction."""
    
    def __init__(
        self,
        data_path: Union[str, Path],
        config: Config,
        split: str = "train",
        transform: Optional[callable] = None
    ):
        """
        Initialize MoA dataset.
        
        Args:
            data_path: Path to processed data directory
            config: Configuration object
            split: Data split ('train', 'val', 'test')
            transform: Optional data transformation
        """
        self.data_path = Path(data_path)
        self.config = config
        self.split = split
        self.transform = transform
        
        # Load data
        self._load_data()
        
        logger.info(f"Loaded {split} dataset: {len(self)} samples")
    
    def _load_data(self):
        """Load dataset from files."""
        split_dir = self.data_path / "splits" / self.split
        
        # Load molecular graphs
        self.molecular_graphs = self._load_molecular_graphs(split_dir)
        
        # Load biological features
        self.biological_features = self._load_biological_features(split_dir)
        
        # Load targets
        self.targets = self._load_targets(split_dir)
        
        # Load metadata
        self.metadata = self._load_metadata(split_dir)
        
        # Validate data consistency
        self._validate_data()
    
    def _load_molecular_graphs(self, split_dir: Path) -> List[Data]:
        """Load molecular graph data."""
        graphs_file = split_dir / "molecular_graphs.pt"
        
        if graphs_file.exists():
            graphs = torch.load(graphs_file)
            logger.debug(f"Loaded {len(graphs)} molecular graphs")
            return graphs
        else:
            logger.warning(f"Molecular graphs file not found: {graphs_file}")
            return []
    
    def _load_biological_features(self, split_dir: Path) -> Dict[str, torch.Tensor]:
        """Load biological features."""
        bio_features = {}
        
        # Mechanism tokens
        mechtoken_file = split_dir / "mechtoken_features.pt"
        if mechtoken_file.exists():
            bio_features['mechtoken_features'] = torch.load(mechtoken_file)
        
        # Gene signatures
        gene_sig_file = split_dir / "gene_signature_features.pt"
        if gene_sig_file.exists():
            bio_features['gene_signature_features'] = torch.load(gene_sig_file)
        
        # Pathway scores
        pathway_file = split_dir / "pathway_score_features.pt"
        if pathway_file.exists():
            bio_features['pathway_score_features'] = torch.load(pathway_file)
        
        logger.debug(f"Loaded biological features: {list(bio_features.keys())}")
        return bio_features
    
    def _load_targets(self, split_dir: Path) -> torch.Tensor:
        """Load target labels."""
        targets_file = split_dir / "targets.pt"
        
        if targets_file.exists():
            targets = torch.load(targets_file)
            logger.debug(f"Loaded targets: {targets.shape}")
            return targets
        else:
            raise FileNotFoundError(f"Targets file not found: {targets_file}")
    
    def _load_metadata(self, split_dir: Path) -> Optional[pd.DataFrame]:
        """Load metadata."""
        metadata_file = split_dir / "metadata.csv"
        
        if metadata_file.exists():
            metadata = pd.read_csv(metadata_file)
            logger.debug(f"Loaded metadata: {metadata.shape}")
            return metadata
        else:
            logger.debug("No metadata file found")
            return None
    
    def _validate_data(self):
        """Validate data consistency."""
        # Check that all modalities have the same number of samples
        num_samples = len(self.targets)
        
        if self.molecular_graphs and len(self.molecular_graphs) != num_samples:
            raise ValueError(f"Molecular graphs count mismatch: {len(self.molecular_graphs)} vs {num_samples}")
        
        for feature_name, features in self.biological_features.items():
            if len(features) != num_samples:
                raise ValueError(f"{feature_name} count mismatch: {len(features)} vs {num_samples}")
        
        if self.metadata is not None and len(self.metadata) != num_samples:
            raise ValueError(f"Metadata count mismatch: {len(self.metadata)} vs {num_samples}")
        
        logger.debug("Data validation passed")
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.targets)
    
    def __getitem__(self, idx: int) -> Tuple[Dict[str, Any], torch.Tensor]:
        """Get a single sample."""
        # Prepare batch data
        batch_data = {}
        
        # Add molecular graph
        if self.molecular_graphs:
            batch_data['molecular_graphs'] = self.molecular_graphs[idx]
        
        # Add biological features
        for feature_name, features in self.biological_features.items():
            batch_data[feature_name] = features[idx]
        
        # Get target
        target = self.targets[idx]
        
        # Apply transform if provided
        if self.transform:
            batch_data, target = self.transform(batch_data, target)
        
        return batch_data, target
    
    def get_sample_metadata(self, idx: int) -> Dict[str, Any]:
        """Get metadata for a specific sample."""
        if self.metadata is not None:
            return self.metadata.iloc[idx].to_dict()
        else:
            return {}
    
    def get_class_weights(self) -> torch.Tensor:
        """Compute class weights for imbalanced learning."""
        # Compute positive class frequencies
        pos_frequencies = self.targets.mean(dim=0)
        
        # Compute weights (inverse frequency)
        weights = 1.0 / (pos_frequencies + 1e-8)
        
        # Normalize weights
        weights = weights / weights.mean()
        
        return weights


class BatchCollator:
    """Custom collate function for MoA prediction batches."""
    
    def __init__(self, config: Config):
        """
        Initialize batch collator.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.enabled_modalities = config.get("scope.modalities", {})
    
    def __call__(self, batch: List[Tuple[Dict[str, Any], torch.Tensor]]) -> Tuple[Dict[str, Any], torch.Tensor]:
        """
        Collate batch of samples.
        
        Args:
            batch: List of (batch_data, target) tuples
            
        Returns:
            Collated batch data and targets
        """
        batch_data_list, targets_list = zip(*batch)
        
        # Collate targets
        targets = torch.stack(targets_list, dim=0)
        
        # Collate batch data
        collated_batch_data = {}
        
        # Handle molecular graphs
        if 'molecular_graphs' in batch_data_list[0]:
            graphs = [data['molecular_graphs'] for data in batch_data_list]
            collated_batch_data['molecular_graphs'] = Batch.from_data_list(graphs)
        
        # Handle biological features
        for feature_name in ['mechtoken_features', 'gene_signature_features', 'pathway_score_features']:
            if feature_name in batch_data_list[0]:
                features = [data[feature_name] for data in batch_data_list]
                collated_batch_data[feature_name] = torch.stack(features, dim=0)
        
        # Handle protein structure features (if enabled)
        if self.enabled_modalities.get('structures', False) and 'protein_features' in batch_data_list[0]:
            protein_features = [data['protein_features'] for data in batch_data_list]
            collated_batch_data['protein_features'] = torch.stack(protein_features, dim=0)
        
        return collated_batch_data, targets


class MoADataLoader:
    """Factory for creating MoA data loaders."""
    
    @staticmethod
    def create_data_loaders(
        data_path: Union[str, Path],
        config: Config,
        splits: List[str] = ["train", "val", "test"]
    ) -> Dict[str, DataLoader]:
        """
        Create data loaders for all splits.
        
        Args:
            data_path: Path to processed data directory
            config: Configuration object
            splits: List of data splits to create loaders for
            
        Returns:
            Dictionary of data loaders
        """
        data_loaders = {}
        collate_fn = BatchCollator(config)
        
        for split in splits:
            # Create dataset
            dataset = MoADataset(data_path, config, split)
            
            # Determine batch size and other parameters
            if split == "train":
                batch_size = config.get("training.batch_size", 32)
                shuffle = True
                drop_last = True
            else:
                batch_size = config.get("training.eval_batch_size", 64)
                shuffle = False
                drop_last = False
            
            # Create data loader
            data_loader = DataLoader(
                dataset=dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                drop_last=drop_last,
                num_workers=config.get("training.num_workers", 4),
                pin_memory=config.get("training.pin_memory", True),
                collate_fn=collate_fn,
                persistent_workers=config.get("training.persistent_workers", True)
            )
            
            data_loaders[split] = data_loader
            
            logger.info(f"Created {split} data loader: {len(dataset)} samples, batch_size={batch_size}")
        
        return data_loaders
    
    @staticmethod
    def create_weighted_sampler(
        dataset: MoADataset,
        config: Config
    ) -> Optional[Sampler]:
        """
        Create weighted sampler for imbalanced datasets.
        
        Args:
            dataset: MoA dataset
            config: Configuration object
            
        Returns:
            Weighted sampler or None
        """
        use_weighted_sampling = config.get("training.use_weighted_sampling", False)
        
        if not use_weighted_sampling:
            return None
        
        # Compute sample weights based on label frequencies
        class_weights = dataset.get_class_weights()
        
        # Compute sample weights
        sample_weights = []
        for i in range(len(dataset)):
            _, target = dataset[i]
            # Weight based on positive labels
            positive_labels = torch.where(target > 0)[0]
            if len(positive_labels) > 0:
                sample_weight = class_weights[positive_labels].mean().item()
            else:
                sample_weight = 1.0  # Default weight for samples with no positive labels
            sample_weights.append(sample_weight)
        
        # Create weighted sampler
        from torch.utils.data import WeightedRandomSampler
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        
        logger.info(f"Created weighted sampler with {len(sample_weights)} samples")
        return sampler


class DataAugmentation:
    """Data augmentation for MoA prediction."""
    
    def __init__(self, config: Config):
        """
        Initialize data augmentation.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.enabled = config.get("training.data_augmentation.enable", False)
        self.noise_std = config.get("training.data_augmentation.noise_std", 0.1)
        self.dropout_prob = config.get("training.data_augmentation.dropout_prob", 0.1)
        
    def __call__(
        self,
        batch_data: Dict[str, Any],
        target: torch.Tensor
    ) -> Tuple[Dict[str, Any], torch.Tensor]:
        """
        Apply data augmentation.
        
        Args:
            batch_data: Batch data dictionary
            target: Target tensor
            
        Returns:
            Augmented batch data and target
        """
        if not self.enabled:
            return batch_data, target
        
        augmented_batch_data = {}
        
        # Augment biological features
        for feature_name in ['mechtoken_features', 'gene_signature_features', 'pathway_score_features']:
            if feature_name in batch_data:
                features = batch_data[feature_name]
                
                # Add Gaussian noise
                if self.noise_std > 0:
                    noise = torch.randn_like(features) * self.noise_std
                    features = features + noise
                
                # Feature dropout
                if self.dropout_prob > 0:
                    dropout_mask = torch.rand_like(features) > self.dropout_prob
                    features = features * dropout_mask
                
                augmented_batch_data[feature_name] = features
            else:
                augmented_batch_data[feature_name] = batch_data[feature_name]
        
        # Copy other data without augmentation
        for key, value in batch_data.items():
            if key not in augmented_batch_data:
                augmented_batch_data[key] = value
        
        return augmented_batch_data, target
