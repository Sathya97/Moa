"""
Curriculum learning for MoA prediction.

This module implements curriculum learning strategies to improve training
by gradually increasing the difficulty of training samples.
"""

import math
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Sampler
import numpy as np
from sklearn.metrics import pairwise_distances

from moa.utils.config import Config
from moa.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class DifficultyScore:
    """Container for sample difficulty scores."""
    sample_idx: int
    difficulty: float
    metadata: Dict


class DifficultyScorer:
    """Compute difficulty scores for training samples."""
    
    def __init__(self, config: Config):
        """
        Initialize difficulty scorer.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.scoring_method = config.get("training.curriculum_learning.scoring_method", "label_frequency")
        
    def compute_difficulty_scores(
        self,
        dataset,
        model: Optional[nn.Module] = None
    ) -> List[DifficultyScore]:
        """
        Compute difficulty scores for all samples in dataset.
        
        Args:
            dataset: Training dataset
            model: Optional pre-trained model for prediction-based scoring
            
        Returns:
            List of difficulty scores
        """
        if self.scoring_method == "label_frequency":
            return self._score_by_label_frequency(dataset)
        elif self.scoring_method == "label_complexity":
            return self._score_by_label_complexity(dataset)
        elif self.scoring_method == "molecular_complexity":
            return self._score_by_molecular_complexity(dataset)
        elif self.scoring_method == "prediction_confidence":
            if model is None:
                raise ValueError("Model required for prediction-based scoring")
            return self._score_by_prediction_confidence(dataset, model)
        elif self.scoring_method == "loss_based":
            if model is None:
                raise ValueError("Model required for loss-based scoring")
            return self._score_by_loss(dataset, model)
        else:
            raise ValueError(f"Unknown scoring method: {self.scoring_method}")
    
    def _score_by_label_frequency(self, dataset) -> List[DifficultyScore]:
        """Score samples by label frequency (rare labels = harder)."""
        # Compute label frequencies
        all_labels = []
        for i in range(len(dataset)):
            _, labels = dataset[i]
            all_labels.append(labels.numpy())
        
        all_labels = np.array(all_labels)
        label_frequencies = np.mean(all_labels, axis=0)
        
        # Compute difficulty scores
        difficulty_scores = []
        for i in range(len(dataset)):
            _, labels = dataset[i]
            labels_np = labels.numpy()
            
            # Average inverse frequency of positive labels
            positive_labels = np.where(labels_np > 0)[0]
            if len(positive_labels) > 0:
                avg_inv_freq = np.mean(1.0 / (label_frequencies[positive_labels] + 1e-8))
                difficulty = min(avg_inv_freq / 10.0, 1.0)  # Normalize
            else:
                difficulty = 0.0  # No positive labels = easy
            
            difficulty_scores.append(DifficultyScore(
                sample_idx=i,
                difficulty=difficulty,
                metadata={'num_positive_labels': len(positive_labels)}
            ))
        
        return difficulty_scores
    
    def _score_by_label_complexity(self, dataset) -> List[DifficultyScore]:
        """Score samples by label complexity (more labels = harder)."""
        difficulty_scores = []
        
        for i in range(len(dataset)):
            _, labels = dataset[i]
            labels_np = labels.numpy()
            
            # Number of positive labels as difficulty
            num_positive = np.sum(labels_np > 0)
            max_labels = len(labels_np)
            difficulty = num_positive / max_labels
            
            difficulty_scores.append(DifficultyScore(
                sample_idx=i,
                difficulty=difficulty,
                metadata={'num_positive_labels': num_positive}
            ))
        
        return difficulty_scores
    
    def _score_by_molecular_complexity(self, dataset) -> List[DifficultyScore]:
        """Score samples by molecular complexity."""
        from rdkit import Chem
        from rdkit.Chem import Descriptors, rdMolDescriptors
        
        difficulty_scores = []
        
        for i in range(len(dataset)):
            batch_data, _ = dataset[i]
            
            # Extract SMILES if available
            smiles = None
            if 'smiles' in batch_data:
                smiles = batch_data['smiles']
            elif hasattr(dataset, 'get_smiles'):
                smiles = dataset.get_smiles(i)
            
            if smiles:
                try:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol:
                        # Compute molecular complexity metrics
                        mw = Descriptors.MolWt(mol)
                        logp = Descriptors.MolLogP(mol)
                        tpsa = Descriptors.TPSA(mol)
                        num_rings = rdMolDescriptors.CalcNumRings(mol)
                        num_rotatable = rdMolDescriptors.CalcNumRotatableBonds(mol)
                        
                        # Combine into difficulty score
                        complexity = (
                            min(mw / 500.0, 1.0) * 0.3 +
                            min(abs(logp) / 5.0, 1.0) * 0.2 +
                            min(tpsa / 150.0, 1.0) * 0.2 +
                            min(num_rings / 5.0, 1.0) * 0.15 +
                            min(num_rotatable / 10.0, 1.0) * 0.15
                        )
                        
                        difficulty = complexity
                    else:
                        difficulty = 0.5  # Default for invalid molecules
                except:
                    difficulty = 0.5  # Default for parsing errors
            else:
                difficulty = 0.5  # Default when SMILES not available
            
            difficulty_scores.append(DifficultyScore(
                sample_idx=i,
                difficulty=difficulty,
                metadata={'smiles': smiles}
            ))
        
        return difficulty_scores
    
    def _score_by_prediction_confidence(
        self,
        dataset,
        model: nn.Module
    ) -> List[DifficultyScore]:
        """Score samples by model prediction confidence (low confidence = harder)."""
        model.eval()
        difficulty_scores = []
        
        with torch.no_grad():
            for i in range(len(dataset)):
                batch_data, labels = dataset[i]
                
                # Add batch dimension
                if isinstance(batch_data, dict):
                    batch_data = {k: v.unsqueeze(0) if hasattr(v, 'unsqueeze') else v 
                                 for k, v in batch_data.items()}
                
                # Get model predictions
                try:
                    predictions = model.predict(batch_data, return_probabilities=True)
                    predictions = predictions.squeeze(0)  # Remove batch dimension
                    
                    # Compute confidence as entropy
                    probs = torch.clamp(predictions, 1e-8, 1-1e-8)
                    entropy = -torch.sum(probs * torch.log(probs) + (1-probs) * torch.log(1-probs))
                    
                    # Normalize entropy to [0, 1]
                    max_entropy = len(predictions) * math.log(2)
                    difficulty = entropy.item() / max_entropy
                    
                except Exception as e:
                    logger.warning(f"Error computing prediction for sample {i}: {e}")
                    difficulty = 0.5  # Default difficulty
                
                difficulty_scores.append(DifficultyScore(
                    sample_idx=i,
                    difficulty=difficulty,
                    metadata={'prediction_entropy': difficulty}
                ))
        
        return difficulty_scores
    
    def _score_by_loss(self, dataset, model: nn.Module) -> List[DifficultyScore]:
        """Score samples by model loss (high loss = harder)."""
        model.eval()
        difficulty_scores = []
        
        with torch.no_grad():
            for i in range(len(dataset)):
                batch_data, labels = dataset[i]
                
                # Add batch dimension
                if isinstance(batch_data, dict):
                    batch_data = {k: v.unsqueeze(0) if hasattr(v, 'unsqueeze') else v 
                                 for k, v in batch_data.items()}
                labels = labels.unsqueeze(0)
                
                # Compute loss
                try:
                    loss = model.compute_loss(batch_data, labels)
                    difficulty = min(loss.item() / 10.0, 1.0)  # Normalize
                except Exception as e:
                    logger.warning(f"Error computing loss for sample {i}: {e}")
                    difficulty = 0.5  # Default difficulty
                
                difficulty_scores.append(DifficultyScore(
                    sample_idx=i,
                    difficulty=difficulty,
                    metadata={'loss': difficulty}
                ))
        
        return difficulty_scores


class CurriculumSampler(Sampler):
    """Sampler that implements curriculum learning."""
    
    def __init__(
        self,
        difficulty_scores: List[DifficultyScore],
        curriculum_strategy: str = "linear",
        epoch: int = 0,
        total_epochs: int = 100
    ):
        """
        Initialize curriculum sampler.
        
        Args:
            difficulty_scores: List of difficulty scores for all samples
            curriculum_strategy: Strategy for curriculum progression
            epoch: Current epoch
            total_epochs: Total number of training epochs
        """
        self.difficulty_scores = difficulty_scores
        self.curriculum_strategy = curriculum_strategy
        self.epoch = epoch
        self.total_epochs = total_epochs
        
        # Sort samples by difficulty
        self.sorted_indices = sorted(
            range(len(difficulty_scores)),
            key=lambda i: difficulty_scores[i].difficulty
        )
    
    def __iter__(self):
        """Generate sample indices according to curriculum."""
        if self.curriculum_strategy == "linear":
            return self._linear_curriculum()
        elif self.curriculum_strategy == "exponential":
            return self._exponential_curriculum()
        elif self.curriculum_strategy == "step":
            return self._step_curriculum()
        elif self.curriculum_strategy == "mixed":
            return self._mixed_curriculum()
        else:
            # Random sampling (no curriculum)
            indices = list(range(len(self.difficulty_scores)))
            np.random.shuffle(indices)
            return iter(indices)
    
    def __len__(self):
        """Return number of samples."""
        return len(self.difficulty_scores)
    
    def _linear_curriculum(self):
        """Linear curriculum: gradually include harder samples."""
        progress = min(self.epoch / self.total_epochs, 1.0)
        num_samples = int(len(self.sorted_indices) * (0.3 + 0.7 * progress))
        
        # Include easiest samples up to current difficulty threshold
        selected_indices = self.sorted_indices[:num_samples]
        np.random.shuffle(selected_indices)
        
        return iter(selected_indices)
    
    def _exponential_curriculum(self):
        """Exponential curriculum: rapid initial growth, then slower."""
        progress = min(self.epoch / self.total_epochs, 1.0)
        difficulty_threshold = 1.0 - math.exp(-3 * progress)
        
        # Include samples below difficulty threshold
        selected_indices = [
            idx for idx in self.sorted_indices
            if self.difficulty_scores[idx].difficulty <= difficulty_threshold
        ]
        
        np.random.shuffle(selected_indices)
        return iter(selected_indices)
    
    def _step_curriculum(self):
        """Step curriculum: discrete difficulty levels."""
        num_steps = 4
        step_size = self.total_epochs // num_steps
        current_step = min(self.epoch // step_size, num_steps - 1)
        
        # Include samples up to current step
        samples_per_step = len(self.sorted_indices) // num_steps
        end_idx = (current_step + 1) * samples_per_step
        selected_indices = self.sorted_indices[:end_idx]
        
        np.random.shuffle(selected_indices)
        return iter(selected_indices)
    
    def _mixed_curriculum(self):
        """Mixed curriculum: combine easy and hard samples."""
        progress = min(self.epoch / self.total_epochs, 1.0)
        
        # Always include some easy samples
        easy_ratio = 0.7 - 0.3 * progress
        hard_ratio = 1.0 - easy_ratio
        
        num_easy = int(len(self.sorted_indices) * easy_ratio)
        num_hard = int(len(self.sorted_indices) * hard_ratio)
        
        # Select easy and hard samples
        easy_indices = self.sorted_indices[:num_easy]
        hard_indices = self.sorted_indices[-num_hard:] if num_hard > 0 else []
        
        selected_indices = easy_indices + hard_indices
        np.random.shuffle(selected_indices)
        
        return iter(selected_indices)


class CurriculumLearning:
    """Main curriculum learning coordinator."""
    
    def __init__(self, config: Config):
        """
        Initialize curriculum learning.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.enabled = config.get("training.curriculum_learning.enable", False)
        self.strategy = config.get("training.curriculum_learning.strategy", "linear")
        self.scoring_method = config.get("training.curriculum_learning.scoring_method", "label_frequency")
        self.update_frequency = config.get("training.curriculum_learning.update_frequency", 10)
        
        self.difficulty_scorer = DifficultyScorer(config)
        self.difficulty_scores = None
        self.last_update_epoch = -1
        
        logger.info(f"Curriculum learning: {self.enabled}")
        if self.enabled:
            logger.info(f"  Strategy: {self.strategy}")
            logger.info(f"  Scoring method: {self.scoring_method}")
            logger.info(f"  Update frequency: {self.update_frequency} epochs")
    
    def initialize_curriculum(self, dataset, model: Optional[nn.Module] = None):
        """Initialize curriculum with difficulty scores."""
        if not self.enabled:
            return
        
        logger.info("Computing initial difficulty scores...")
        self.difficulty_scores = self.difficulty_scorer.compute_difficulty_scores(
            dataset, model
        )
        
        # Log difficulty statistics
        difficulties = [score.difficulty for score in self.difficulty_scores]
        logger.info(f"Difficulty scores computed:")
        logger.info(f"  Mean: {np.mean(difficulties):.3f}")
        logger.info(f"  Std: {np.std(difficulties):.3f}")
        logger.info(f"  Min: {np.min(difficulties):.3f}")
        logger.info(f"  Max: {np.max(difficulties):.3f}")
    
    def get_curriculum_loader(
        self,
        original_loader: DataLoader,
        epoch: int,
        model: Optional[nn.Module] = None
    ) -> DataLoader:
        """Get curriculum-based data loader for current epoch."""
        if not self.enabled or self.difficulty_scores is None:
            return original_loader
        
        # Update difficulty scores periodically
        if (epoch - self.last_update_epoch) >= self.update_frequency and model is not None:
            logger.info(f"Updating difficulty scores at epoch {epoch}")
            self.difficulty_scores = self.difficulty_scorer.compute_difficulty_scores(
                original_loader.dataset, model
            )
            self.last_update_epoch = epoch
        
        # Create curriculum sampler
        curriculum_sampler = CurriculumSampler(
            difficulty_scores=self.difficulty_scores,
            curriculum_strategy=self.strategy,
            epoch=epoch,
            total_epochs=self.config.get("training.num_epochs", 100)
        )
        
        # Create new data loader with curriculum sampler
        curriculum_loader = DataLoader(
            dataset=original_loader.dataset,
            batch_size=original_loader.batch_size,
            sampler=curriculum_sampler,
            num_workers=original_loader.num_workers,
            collate_fn=original_loader.collate_fn,
            pin_memory=original_loader.pin_memory
        )
        
        return curriculum_loader
    
    def get_difficulty_statistics(self) -> Dict:
        """Get statistics about current difficulty scores."""
        if self.difficulty_scores is None:
            return {}
        
        difficulties = [score.difficulty for score in self.difficulty_scores]
        
        return {
            'mean_difficulty': np.mean(difficulties),
            'std_difficulty': np.std(difficulties),
            'min_difficulty': np.min(difficulties),
            'max_difficulty': np.max(difficulties),
            'num_samples': len(difficulties)
        }
