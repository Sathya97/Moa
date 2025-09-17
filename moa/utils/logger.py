"""Logging utilities for the MoA prediction framework."""

import logging
import sys
from pathlib import Path
from typing import Optional, Union

import wandb
from pytorch_lightning.loggers import WandbLogger


def setup_logging(
    level: Union[str, int] = logging.INFO,
    log_file: Optional[Union[str, Path]] = None,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Setup logging configuration.
    
    Args:
        level: Logging level
        log_file: Optional log file path
        format_string: Custom format string
        
    Returns:
        Configured logger
    """
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Configure root logger
    logging.basicConfig(
        level=level,
        format=format_string,
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger("moa")
    
    # Add file handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter(format_string))
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str = "moa") -> logging.Logger:
    """
    Get a logger instance.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


def setup_wandb_logger(
    project: str,
    entity: Optional[str] = None,
    name: Optional[str] = None,
    config: Optional[dict] = None,
    tags: Optional[list] = None,
    notes: Optional[str] = None,
    offline: bool = False
) -> WandbLogger:
    """
    Setup Weights & Biases logger for PyTorch Lightning.
    
    Args:
        project: W&B project name
        entity: W&B entity (team/user)
        name: Run name
        config: Configuration dictionary to log
        tags: List of tags for the run
        notes: Notes for the run
        offline: Whether to run in offline mode
        
    Returns:
        WandbLogger instance
    """
    wandb_logger = WandbLogger(
        project=project,
        entity=entity,
        name=name,
        config=config,
        tags=tags,
        notes=notes,
        offline=offline,
        log_model=True,
        save_dir="wandb_logs"
    )
    
    return wandb_logger


def log_model_summary(model, logger: logging.Logger) -> None:
    """
    Log model summary information.
    
    Args:
        model: PyTorch model
        logger: Logger instance
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"Model Summary:")
    logger.info(f"  Total parameters: {total_params:,}")
    logger.info(f"  Trainable parameters: {trainable_params:,}")
    logger.info(f"  Non-trainable parameters: {total_params - trainable_params:,}")


def log_config(config: dict, logger: logging.Logger) -> None:
    """
    Log configuration information.
    
    Args:
        config: Configuration dictionary
        logger: Logger instance
    """
    logger.info("Configuration:")
    for key, value in config.items():
        if isinstance(value, dict):
            logger.info(f"  {key}:")
            for sub_key, sub_value in value.items():
                logger.info(f"    {sub_key}: {sub_value}")
        else:
            logger.info(f"  {key}: {value}")


class MetricsLogger:
    """Utility class for logging training metrics."""
    
    def __init__(self, logger: logging.Logger, wandb_logger: Optional[WandbLogger] = None):
        """
        Initialize metrics logger.
        
        Args:
            logger: Python logger
            wandb_logger: Optional W&B logger
        """
        self.logger = logger
        self.wandb_logger = wandb_logger
    
    def log_metrics(self, metrics: dict, step: Optional[int] = None, prefix: str = "") -> None:
        """
        Log metrics to both Python logger and W&B.
        
        Args:
            metrics: Dictionary of metrics
            step: Optional step number
            prefix: Optional prefix for metric names
        """
        # Log to Python logger
        metric_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        if prefix:
            metric_str = f"{prefix} - {metric_str}"
        if step is not None:
            metric_str = f"Step {step} - {metric_str}"
        
        self.logger.info(metric_str)
        
        # Log to W&B
        if self.wandb_logger:
            wandb_metrics = {f"{prefix}_{k}" if prefix else k: v for k, v in metrics.items()}
            self.wandb_logger.log_metrics(wandb_metrics, step=step)
    
    def log_epoch_metrics(self, epoch: int, train_metrics: dict, val_metrics: dict) -> None:
        """
        Log epoch-level metrics.
        
        Args:
            epoch: Epoch number
            train_metrics: Training metrics
            val_metrics: Validation metrics
        """
        self.log_metrics(train_metrics, step=epoch, prefix="train")
        self.log_metrics(val_metrics, step=epoch, prefix="val")
        
        # Log epoch summary
        train_loss = train_metrics.get("loss", 0)
        val_loss = val_metrics.get("loss", 0)
        val_f1 = val_metrics.get("macro_f1", 0)
        
        self.logger.info(
            f"Epoch {epoch} Summary - "
            f"Train Loss: {train_loss:.4f}, "
            f"Val Loss: {val_loss:.4f}, "
            f"Val F1: {val_f1:.4f}"
        )
