"""
Training monitoring and logging utilities.

This module provides comprehensive monitoring of training progress,
metrics logging, and visualization tools.
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from collections import defaultdict, deque

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorboardX import SummaryWriter

from moa.utils.config import Config
from moa.utils.logger import get_logger

logger = get_logger(__name__)


class MetricsLogger:
    """Logger for training and validation metrics."""
    
    def __init__(self, log_dir: Union[str, Path]):
        """
        Initialize metrics logger.
        
        Args:
            log_dir: Directory to save logs
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize storage
        self.metrics_history = defaultdict(list)
        self.batch_metrics = defaultdict(list)
        
        # File paths
        self.metrics_file = self.log_dir / "metrics.json"
        self.batch_metrics_file = self.log_dir / "batch_metrics.json"
        
        logger.info(f"Initialized metrics logger: {self.log_dir}")
    
    def log_epoch_metrics(self, metrics: Dict[str, float], epoch: int):
        """Log epoch-level metrics."""
        # Add timestamp and epoch
        metrics_with_meta = {
            'epoch': epoch,
            'timestamp': time.time(),
            **metrics
        }
        
        # Store in memory
        for key, value in metrics_with_meta.items():
            self.metrics_history[key].append(value)
        
        # Save to file
        self._save_metrics()
        
        logger.debug(f"Logged epoch {epoch} metrics: {len(metrics)} metrics")
    
    def log_batch_metrics(self, metrics: Dict[str, float]):
        """Log batch-level metrics."""
        # Add timestamp
        metrics_with_meta = {
            'timestamp': time.time(),
            **metrics
        }
        
        # Store in memory (keep only recent batches)
        for key, value in metrics_with_meta.items():
            if len(self.batch_metrics[key]) > 10000:  # Limit memory usage
                self.batch_metrics[key] = self.batch_metrics[key][-5000:]
            self.batch_metrics[key].append(value)
        
        # Save periodically
        if len(self.batch_metrics['timestamp']) % 100 == 0:
            self._save_batch_metrics()
    
    def get_metrics_dataframe(self) -> pd.DataFrame:
        """Get metrics as pandas DataFrame."""
        if not self.metrics_history:
            return pd.DataFrame()
        
        return pd.DataFrame(self.metrics_history)
    
    def get_latest_metrics(self) -> Dict[str, float]:
        """Get latest epoch metrics."""
        if not self.metrics_history:
            return {}
        
        latest_metrics = {}
        for key, values in self.metrics_history.items():
            if values:
                latest_metrics[key] = values[-1]
        
        return latest_metrics
    
    def _save_metrics(self):
        """Save metrics to file."""
        try:
            with open(self.metrics_file, 'w') as f:
                json.dump(dict(self.metrics_history), f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save metrics: {e}")
    
    def _save_batch_metrics(self):
        """Save batch metrics to file."""
        try:
            with open(self.batch_metrics_file, 'w') as f:
                json.dump(dict(self.batch_metrics), f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save batch metrics: {e}")
    
    def load_metrics(self):
        """Load metrics from file."""
        try:
            if self.metrics_file.exists():
                with open(self.metrics_file, 'r') as f:
                    loaded_metrics = json.load(f)
                    self.metrics_history = defaultdict(list, loaded_metrics)
                logger.info("Loaded existing metrics")
        except Exception as e:
            logger.warning(f"Failed to load metrics: {e}")


class TensorBoardLogger:
    """TensorBoard logging wrapper."""
    
    def __init__(self, log_dir: Union[str, Path]):
        """
        Initialize TensorBoard logger.
        
        Args:
            log_dir: Directory for TensorBoard logs
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            self.writer = SummaryWriter(str(self.log_dir))
            self.enabled = True
            logger.info(f"Initialized TensorBoard logger: {self.log_dir}")
        except Exception as e:
            logger.warning(f"Failed to initialize TensorBoard: {e}")
            self.writer = None
            self.enabled = False
    
    def log_scalar(self, tag: str, value: float, step: int):
        """Log scalar value."""
        if self.enabled:
            self.writer.add_scalar(tag, value, step)
    
    def log_scalars(self, tag: str, values: Dict[str, float], step: int):
        """Log multiple scalar values."""
        if self.enabled:
            self.writer.add_scalars(tag, values, step)
    
    def log_histogram(self, tag: str, values: torch.Tensor, step: int):
        """Log histogram of values."""
        if self.enabled:
            self.writer.add_histogram(tag, values, step)
    
    def log_image(self, tag: str, image: torch.Tensor, step: int):
        """Log image."""
        if self.enabled:
            self.writer.add_image(tag, image, step)
    
    def log_figure(self, tag: str, figure: plt.Figure, step: int):
        """Log matplotlib figure."""
        if self.enabled:
            self.writer.add_figure(tag, figure, step)
    
    def close(self):
        """Close TensorBoard writer."""
        if self.enabled:
            self.writer.close()


class TrainingMonitor:
    """Comprehensive training monitoring."""
    
    def __init__(self, config: Config):
        """
        Initialize training monitor.
        
        Args:
            config: Configuration object
        """
        self.config = config
        
        # Setup logging directories
        self.log_dir = Path(config.get("training.log_dir", "logs"))
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize loggers
        self.metrics_logger = MetricsLogger(self.log_dir / "metrics")
        self.tensorboard_logger = TensorBoardLogger(self.log_dir / "tensorboard")
        
        # Monitoring configuration
        self.log_frequency = config.get("training.log_frequency", 10)
        self.plot_frequency = config.get("training.plot_frequency", 50)
        self.save_plots = config.get("training.save_plots", True)
        
        # Performance tracking
        self.training_start_time = None
        self.epoch_times = deque(maxlen=100)
        self.best_metrics = {}
        
        logger.info(f"Initialized training monitor: {self.log_dir}")
    
    def start_training(self):
        """Mark start of training."""
        self.training_start_time = time.time()
        logger.info("Training monitoring started")
    
    def log_epoch_metrics(self, metrics: Dict[str, float], epoch: int):
        """Log epoch metrics to all loggers."""
        # Add timing information
        if self.training_start_time:
            metrics['elapsed_time'] = time.time() - self.training_start_time
        
        # Log to metrics logger
        self.metrics_logger.log_epoch_metrics(metrics, epoch)
        
        # Log to TensorBoard
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self.tensorboard_logger.log_scalar(f"epoch/{key}", value, epoch)
        
        # Update best metrics
        self._update_best_metrics(metrics)
        
        # Generate plots periodically
        if epoch % self.plot_frequency == 0:
            self._generate_training_plots(epoch)
    
    def log_batch_metrics(self, metrics: Dict[str, float]):
        """Log batch metrics."""
        self.metrics_logger.log_batch_metrics(metrics)
        
        # Log to TensorBoard
        step = metrics.get('global_step', 0)
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and key != 'global_step':
                self.tensorboard_logger.log_scalar(f"batch/{key}", value, step)
    
    def log_model_parameters(self, model: torch.nn.Module, epoch: int):
        """Log model parameters and gradients."""
        for name, param in model.named_parameters():
            if param.requires_grad:
                # Log parameter values
                self.tensorboard_logger.log_histogram(
                    f"parameters/{name}", param.data, epoch
                )
                
                # Log gradients if available
                if param.grad is not None:
                    self.tensorboard_logger.log_histogram(
                        f"gradients/{name}", param.grad.data, epoch
                    )
    
    def log_learning_rates(self, optimizer: torch.optim.Optimizer, epoch: int):
        """Log learning rates for all parameter groups."""
        for i, param_group in enumerate(optimizer.param_groups):
            lr = param_group['lr']
            group_name = param_group.get('name', f'group_{i}')
            self.tensorboard_logger.log_scalar(f"learning_rate/{group_name}", lr, epoch)
    
    def _update_best_metrics(self, metrics: Dict[str, float]):
        """Update best metrics tracking."""
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                if key not in self.best_metrics:
                    self.best_metrics[key] = value
                else:
                    # Assume higher is better for most metrics, lower for loss
                    if 'loss' in key.lower():
                        self.best_metrics[key] = min(self.best_metrics[key], value)
                    else:
                        self.best_metrics[key] = max(self.best_metrics[key], value)
    
    def _generate_training_plots(self, epoch: int):
        """Generate training progress plots."""
        try:
            df = self.metrics_logger.get_metrics_dataframe()
            if df.empty:
                return
            
            # Create plots
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f'Training Progress - Epoch {epoch}', fontsize=16)
            
            # Loss curves
            if 'train_loss' in df.columns and 'val_loss' in df.columns:
                axes[0, 0].plot(df['epoch'], df['train_loss'], label='Train Loss', alpha=0.7)
                axes[0, 0].plot(df['epoch'], df['val_loss'], label='Val Loss', alpha=0.7)
                axes[0, 0].set_xlabel('Epoch')
                axes[0, 0].set_ylabel('Loss')
                axes[0, 0].set_title('Training and Validation Loss')
                axes[0, 0].legend()
                axes[0, 0].grid(True, alpha=0.3)
            
            # AUROC curves
            auroc_cols = [col for col in df.columns if 'auroc' in col.lower()]
            if auroc_cols:
                for col in auroc_cols:
                    axes[0, 1].plot(df['epoch'], df[col], label=col, alpha=0.7)
                axes[0, 1].set_xlabel('Epoch')
                axes[0, 1].set_ylabel('AUROC')
                axes[0, 1].set_title('AUROC Metrics')
                axes[0, 1].legend()
                axes[0, 1].grid(True, alpha=0.3)
            
            # F1 scores
            f1_cols = [col for col in df.columns if 'f1' in col.lower()]
            if f1_cols:
                for col in f1_cols:
                    axes[1, 0].plot(df['epoch'], df[col], label=col, alpha=0.7)
                axes[1, 0].set_xlabel('Epoch')
                axes[1, 0].set_ylabel('F1 Score')
                axes[1, 0].set_title('F1 Score Metrics')
                axes[1, 0].legend()
                axes[1, 0].grid(True, alpha=0.3)
            
            # Learning rate
            if 'learning_rate' in df.columns:
                axes[1, 1].plot(df['epoch'], df['learning_rate'], alpha=0.7)
                axes[1, 1].set_xlabel('Epoch')
                axes[1, 1].set_ylabel('Learning Rate')
                axes[1, 1].set_title('Learning Rate Schedule')
                axes[1, 1].set_yscale('log')
                axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save plot
            if self.save_plots:
                plot_path = self.log_dir / f"training_progress_epoch_{epoch}.png"
                plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            
            # Log to TensorBoard
            self.tensorboard_logger.log_figure("training_progress", fig, epoch)
            
            plt.close(fig)
            
        except Exception as e:
            logger.warning(f"Failed to generate training plots: {e}")
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get comprehensive training summary."""
        df = self.metrics_logger.get_metrics_dataframe()
        
        summary = {
            'total_epochs': len(df) if not df.empty else 0,
            'best_metrics': self.best_metrics.copy(),
            'latest_metrics': self.metrics_logger.get_latest_metrics(),
            'training_time': time.time() - self.training_start_time if self.training_start_time else 0
        }
        
        # Add convergence analysis
        if not df.empty and 'val_loss' in df.columns:
            val_losses = df['val_loss'].values
            if len(val_losses) > 10:
                # Check if loss is still decreasing
                recent_trend = np.polyfit(range(len(val_losses[-10:])), val_losses[-10:], 1)[0]
                summary['loss_trend'] = 'decreasing' if recent_trend < 0 else 'increasing'
                summary['loss_trend_slope'] = recent_trend
        
        return summary
    
    def close(self):
        """Close all loggers."""
        self.tensorboard_logger.close()
        logger.info("Training monitoring closed")
