"""
Main training class for MoA prediction models.

This module implements the comprehensive training pipeline with support for
multi-modal models, curriculum learning, and advanced optimization strategies.
"""

import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from tqdm import tqdm

from moa.models.multimodal_model import MultiModalMoAPredictor
from moa.training.optimization import OptimizerFactory, SchedulerFactory
from moa.training.monitoring import TrainingMonitor
from moa.training.curriculum import CurriculumLearning
from moa.utils.config import Config
from moa.utils.logger import get_logger

logger = get_logger(__name__)


class MoATrainer:
    """Main trainer class for MoA prediction models."""
    
    def __init__(
        self,
        model: MultiModalMoAPredictor,
        config: Config,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: Optional[DataLoader] = None
    ):
        """
        Initialize MoA trainer.
        
        Args:
            model: Multi-modal MoA prediction model
            config: Configuration object
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Test data loader (optional)
        """
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        
        # Training parameters
        self.num_epochs = config.get("training.num_epochs", 100)
        self.patience = config.get("training.early_stopping.patience", 10)
        self.min_delta = config.get("training.early_stopping.min_delta", 1e-4)
        self.gradient_clip_val = config.get("training.gradient_clip_val", 1.0)
        self.accumulate_grad_batches = config.get("training.accumulate_grad_batches", 1)
        
        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Optimizer and scheduler
        self.optimizer = OptimizerFactory.create_optimizer(model, config)
        self.scheduler = SchedulerFactory.create_scheduler(self.optimizer, config)
        
        # Curriculum learning
        self.use_curriculum = config.get("training.curriculum_learning.enable", False)
        if self.use_curriculum:
            self.curriculum = CurriculumLearning(config)
        else:
            self.curriculum = None
        
        # Training monitoring
        self.monitor = TrainingMonitor(config)
        
        # Model checkpointing
        self.checkpoint_dir = Path(config.get("training.checkpoint_dir", "checkpoints"))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_top_k = config.get("training.save_top_k", 3)
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_score = float('-inf')
        self.patience_counter = 0
        self.training_history = []
        
        logger.info(f"Initialized MoATrainer with {sum(p.numel() for p in model.parameters()):,} parameters")
        logger.info(f"Training on device: {self.device}")
        logger.info(f"Optimizer: {type(self.optimizer).__name__}")
        logger.info(f"Scheduler: {type(self.scheduler).__name__ if self.scheduler else 'None'}")
        logger.info(f"Curriculum learning: {self.use_curriculum}")
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        epoch_metrics = {
            'train_loss': 0.0,
            'train_loss_classification': 0.0,
            'train_loss_prototype': 0.0,
            'train_loss_invariance': 0.0,
            'train_loss_contrastive': 0.0,
            'num_batches': 0
        }
        
        # Get curriculum-ordered batches if using curriculum learning
        if self.curriculum:
            data_loader = self.curriculum.get_curriculum_loader(
                self.train_loader, self.current_epoch
            )
        else:
            data_loader = self.train_loader
        
        progress_bar = tqdm(data_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            batch_data, targets = self._prepare_batch(batch)
            
            # Forward pass and loss computation
            loss, loss_components = self.model.compute_loss(
                batch_data, targets, return_components=True
            )
            
            # Scale loss for gradient accumulation
            loss = loss / self.accumulate_grad_batches
            
            # Backward pass
            loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % self.accumulate_grad_batches == 0:
                # Gradient clipping
                if self.gradient_clip_val > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.gradient_clip_val
                    )
                
                # Optimizer step
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.global_step += 1
                
                # Learning rate scheduling (step-based)
                if self.scheduler and hasattr(self.scheduler, 'step_update'):
                    self.scheduler.step_update(self.global_step)
            
            # Update metrics
            epoch_metrics['train_loss'] += loss.item() * self.accumulate_grad_batches
            for component, value in loss_components.items():
                key = f'train_loss_{component}'
                if key in epoch_metrics:
                    epoch_metrics[key] += value.item()
            epoch_metrics['num_batches'] += 1
            
            # Update progress bar
            current_loss = epoch_metrics['train_loss'] / epoch_metrics['num_batches']
            progress_bar.set_postfix({'loss': f'{current_loss:.4f}'})
            
            # Log batch metrics
            if batch_idx % self.config.get("training.log_every_n_steps", 100) == 0:
                self.monitor.log_batch_metrics({
                    'batch_loss': loss.item() * self.accumulate_grad_batches,
                    'learning_rate': self.optimizer.param_groups[0]['lr'],
                    'epoch': self.current_epoch,
                    'global_step': self.global_step
                })
        
        # Average metrics over epoch
        for key in epoch_metrics:
            if key != 'num_batches':
                epoch_metrics[key] /= epoch_metrics['num_batches']
        
        return epoch_metrics
    
    def validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        val_metrics = {
            'val_loss': 0.0,
            'val_loss_classification': 0.0,
            'val_loss_prototype': 0.0,
            'val_loss_invariance': 0.0,
            'val_loss_contrastive': 0.0,
            'num_batches': 0
        }
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                # Move batch to device
                batch_data, targets = self._prepare_batch(batch)
                
                # Forward pass
                predictions = self.model.predict(batch_data, return_probabilities=True)
                loss, loss_components = self.model.compute_loss(
                    batch_data, targets, return_components=True
                )
                
                # Update metrics
                val_metrics['val_loss'] += loss.item()
                for component, value in loss_components.items():
                    key = f'val_loss_{component}'
                    if key in val_metrics:
                        val_metrics[key] += value.item()
                val_metrics['num_batches'] += 1
                
                # Collect predictions and targets for evaluation
                all_predictions.append(predictions.cpu())
                all_targets.append(targets.cpu())
        
        # Average metrics over epoch
        for key in val_metrics:
            if key != 'num_batches':
                val_metrics[key] /= val_metrics['num_batches']
        
        # Compute evaluation metrics
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        eval_metrics = self._compute_evaluation_metrics(all_predictions, all_targets)
        val_metrics.update(eval_metrics)
        
        return val_metrics
    
    def train(self) -> Dict[str, Any]:
        """Main training loop."""
        logger.info(f"Starting training for {self.num_epochs} epochs")
        
        for epoch in range(self.num_epochs):
            self.current_epoch = epoch
            
            # Training phase
            train_metrics = self.train_epoch()
            
            # Validation phase
            val_metrics = self.validate_epoch()
            
            # Combine metrics
            epoch_metrics = {**train_metrics, **val_metrics}
            
            # Learning rate scheduling (epoch-based)
            if self.scheduler and not hasattr(self.scheduler, 'step_update'):
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['val_loss'])
                else:
                    self.scheduler.step()
            
            # Log epoch metrics
            self.monitor.log_epoch_metrics(epoch_metrics, epoch)
            self.training_history.append(epoch_metrics)
            
            # Model checkpointing
            val_score = val_metrics.get('val_auroc_macro', -val_metrics['val_loss'])
            is_best = val_score > self.best_val_score
            
            if is_best:
                self.best_val_score = val_score
                self.patience_counter = 0
                self._save_checkpoint(epoch, epoch_metrics, is_best=True)
            else:
                self.patience_counter += 1
            
            # Regular checkpointing
            if epoch % self.config.get("training.save_every_n_epochs", 10) == 0:
                self._save_checkpoint(epoch, epoch_metrics, is_best=False)
            
            # Early stopping
            if self.patience_counter >= self.patience:
                logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                break
            
            # Log progress
            logger.info(
                f"Epoch {epoch}: "
                f"train_loss={train_metrics['train_loss']:.4f}, "
                f"val_loss={val_metrics['val_loss']:.4f}, "
                f"val_auroc={val_metrics.get('val_auroc_macro', 0):.4f}, "
                f"lr={self.optimizer.param_groups[0]['lr']:.2e}"
            )
        
        # Final evaluation on test set
        test_metrics = {}
        if self.test_loader:
            test_metrics = self.evaluate(self.test_loader, split_name="test")
        
        # Training summary
        training_summary = {
            'best_val_score': self.best_val_score,
            'total_epochs': self.current_epoch + 1,
            'total_steps': self.global_step,
            'training_history': self.training_history,
            'test_metrics': test_metrics
        }
        
        logger.info(f"Training completed. Best validation score: {self.best_val_score:.4f}")
        
        return training_summary
    
    def evaluate(self, data_loader: DataLoader, split_name: str = "test") -> Dict[str, float]:
        """Evaluate model on given data loader."""
        self.model.eval()
        
        all_predictions = []
        all_targets = []
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc=f"Evaluating {split_name}"):
                batch_data, targets = self._prepare_batch(batch)
                
                # Forward pass
                predictions = self.model.predict(batch_data, return_probabilities=True)
                loss = self.model.compute_loss(batch_data, targets)
                
                # Accumulate results
                all_predictions.append(predictions.cpu())
                all_targets.append(targets.cpu())
                total_loss += loss.item()
                num_batches += 1
        
        # Compute metrics
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        eval_metrics = self._compute_evaluation_metrics(all_predictions, all_targets)
        eval_metrics[f'{split_name}_loss'] = total_loss / num_batches
        
        logger.info(f"{split_name.capitalize()} evaluation completed")
        for metric, value in eval_metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        return eval_metrics
    
    def _prepare_batch(self, batch: Tuple) -> Tuple[Dict, torch.Tensor]:
        """Prepare batch for training/evaluation."""
        batch_data, targets = batch
        
        # Move targets to device
        targets = targets.to(self.device)
        
        # Move batch data to device
        if isinstance(batch_data, dict):
            for key, value in batch_data.items():
                if hasattr(value, 'to'):
                    batch_data[key] = value.to(self.device)
        
        return batch_data, targets
    
    def _compute_evaluation_metrics(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> Dict[str, float]:
        """Compute evaluation metrics."""
        from sklearn.metrics import (
            roc_auc_score, average_precision_score, f1_score,
            precision_score, recall_score, accuracy_score
        )
        
        # Convert to numpy
        y_pred = predictions.numpy()
        y_true = targets.numpy()
        
        metrics = {}
        
        try:
            # AUROC scores
            metrics['val_auroc_macro'] = roc_auc_score(y_true, y_pred, average='macro')
            metrics['val_auroc_micro'] = roc_auc_score(y_true, y_pred, average='micro')
            
            # AUPRC scores
            metrics['val_auprc_macro'] = average_precision_score(y_true, y_pred, average='macro')
            metrics['val_auprc_micro'] = average_precision_score(y_true, y_pred, average='micro')
            
            # Binary predictions for classification metrics
            y_pred_binary = (y_pred > 0.5).astype(int)
            
            # F1 scores
            metrics['val_f1_macro'] = f1_score(y_true, y_pred_binary, average='macro', zero_division=0)
            metrics['val_f1_micro'] = f1_score(y_true, y_pred_binary, average='micro', zero_division=0)
            
            # Precision and Recall
            metrics['val_precision_macro'] = precision_score(y_true, y_pred_binary, average='macro', zero_division=0)
            metrics['val_recall_macro'] = recall_score(y_true, y_pred_binary, average='macro', zero_division=0)
            
            # Subset accuracy (exact match)
            metrics['val_subset_accuracy'] = accuracy_score(y_true, y_pred_binary)
            
        except Exception as e:
            logger.warning(f"Error computing evaluation metrics: {e}")
            # Fallback metrics
            metrics['val_auroc_macro'] = 0.5
            metrics['val_f1_macro'] = 0.0
        
        return metrics
    
    def _save_checkpoint(
        self,
        epoch: int,
        metrics: Dict[str, float],
        is_best: bool = False
    ) -> None:
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_score': self.best_val_score,
            'metrics': metrics,
            'config': self.config.to_dict()
        }
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model checkpoint: {best_path}")
        
        # Clean up old checkpoints (keep only top-k)
        self._cleanup_checkpoints()
    
    def _cleanup_checkpoints(self) -> None:
        """Clean up old checkpoints, keeping only the best ones."""
        checkpoint_files = list(self.checkpoint_dir.glob("checkpoint_epoch_*.pt"))
        
        if len(checkpoint_files) > self.save_top_k:
            # Sort by modification time and keep only the newest
            checkpoint_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            
            for old_checkpoint in checkpoint_files[self.save_top_k:]:
                old_checkpoint.unlink()
    
    def load_checkpoint(self, checkpoint_path: Union[str, Path]) -> None:
        """Load model checkpoint."""
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load training state
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_score = checkpoint['best_val_score']
        
        logger.info(f"Loaded checkpoint from epoch {self.current_epoch}")
        logger.info(f"Best validation score: {self.best_val_score:.4f}")

    def get_model_summary(self) -> Dict[str, Any]:
        """Get comprehensive model summary."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        summary = {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / 1024 / 1024,
            'device': str(self.device),
            'optimizer': type(self.optimizer).__name__,
            'scheduler': type(self.scheduler).__name__ if self.scheduler else None,
            'current_epoch': self.current_epoch,
            'global_step': self.global_step,
            'best_val_score': self.best_val_score
        }

        return summary
