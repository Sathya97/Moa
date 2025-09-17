"""
Optimization utilities for MoA prediction training.

This module provides optimizers, learning rate schedulers, and advanced
optimization strategies for multi-modal model training.
"""

import math
from typing import Dict, Any, Optional, Union

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler

from moa.utils.config import Config
from moa.utils.logger import get_logger

logger = get_logger(__name__)


class OptimizerFactory:
    """Factory for creating optimizers."""
    
    @staticmethod
    def create_optimizer(model: nn.Module, config: Config) -> optim.Optimizer:
        """
        Create optimizer based on configuration.
        
        Args:
            model: Model to optimize
            config: Configuration object
            
        Returns:
            Configured optimizer
        """
        optimizer_name = config.get("training.optimizer.name", "adamw")
        learning_rate = config.get("training.optimizer.learning_rate", 1e-3)
        weight_decay = config.get("training.optimizer.weight_decay", 1e-4)
        
        # Get model parameters with different learning rates for different components
        param_groups = OptimizerFactory._get_parameter_groups(model, config)
        
        if optimizer_name.lower() == "adamw":
            optimizer = optim.AdamW(
                param_groups,
                lr=learning_rate,
                weight_decay=weight_decay,
                betas=config.get("training.optimizer.betas", (0.9, 0.999)),
                eps=config.get("training.optimizer.eps", 1e-8)
            )
        elif optimizer_name.lower() == "adam":
            optimizer = optim.Adam(
                param_groups,
                lr=learning_rate,
                weight_decay=weight_decay,
                betas=config.get("training.optimizer.betas", (0.9, 0.999)),
                eps=config.get("training.optimizer.eps", 1e-8)
            )
        elif optimizer_name.lower() == "sgd":
            optimizer = optim.SGD(
                param_groups,
                lr=learning_rate,
                weight_decay=weight_decay,
                momentum=config.get("training.optimizer.momentum", 0.9),
                nesterov=config.get("training.optimizer.nesterov", True)
            )
        elif optimizer_name.lower() == "rmsprop":
            optimizer = optim.RMSprop(
                param_groups,
                lr=learning_rate,
                weight_decay=weight_decay,
                momentum=config.get("training.optimizer.momentum", 0.9),
                alpha=config.get("training.optimizer.alpha", 0.99)
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
        
        logger.info(f"Created {optimizer_name} optimizer with lr={learning_rate}")
        return optimizer
    
    @staticmethod
    def _get_parameter_groups(model: nn.Module, config: Config) -> list:
        """Get parameter groups with different learning rates."""
        base_lr = config.get("training.optimizer.learning_rate", 1e-3)
        
        # Different learning rates for different components
        lr_multipliers = config.get("training.optimizer.lr_multipliers", {})
        
        param_groups = []
        
        # Graph transformer parameters
        if hasattr(model, 'modality_encoders') and 'chemistry' in model.modality_encoders:
            graph_params = list(model.modality_encoders['chemistry'].parameters())
            graph_lr = base_lr * lr_multipliers.get('graph_transformer', 1.0)
            param_groups.append({
                'params': graph_params,
                'lr': graph_lr,
                'name': 'graph_transformer'
            })
        
        # Pathway transformer parameters
        if hasattr(model, 'modality_encoders') and 'biology' in model.modality_encoders:
            pathway_params = list(model.modality_encoders['biology'].parameters())
            pathway_lr = base_lr * lr_multipliers.get('pathway_transformer', 1.0)
            param_groups.append({
                'params': pathway_params,
                'lr': pathway_lr,
                'name': 'pathway_transformer'
            })
        
        # Fusion layer parameters
        if hasattr(model, 'fusion_layer'):
            fusion_params = list(model.fusion_layer.parameters())
            fusion_lr = base_lr * lr_multipliers.get('fusion_layer', 1.0)
            param_groups.append({
                'params': fusion_params,
                'lr': fusion_lr,
                'name': 'fusion_layer'
            })
        
        # Prediction head parameters
        if hasattr(model, 'prediction_head'):
            head_params = list(model.prediction_head.parameters())
            head_lr = base_lr * lr_multipliers.get('prediction_head', 1.0)
            param_groups.append({
                'params': head_params,
                'lr': head_lr,
                'name': 'prediction_head'
            })
        
        # Loss function parameters (prototypes)
        if hasattr(model, 'loss_fn') and hasattr(model.loss_fn, 'prototype_loss'):
            proto_params = list(model.loss_fn.prototype_loss.parameters())
            proto_lr = base_lr * lr_multipliers.get('prototypes', 1.0)
            param_groups.append({
                'params': proto_params,
                'lr': proto_lr,
                'name': 'prototypes'
            })
        
        # If no specific groups were created, use all parameters
        if not param_groups:
            param_groups = [{'params': model.parameters(), 'lr': base_lr}]
        
        return param_groups


class SchedulerFactory:
    """Factory for creating learning rate schedulers."""
    
    @staticmethod
    def create_scheduler(
        optimizer: optim.Optimizer,
        config: Config
    ) -> Optional[_LRScheduler]:
        """
        Create learning rate scheduler based on configuration.
        
        Args:
            optimizer: Optimizer to schedule
            config: Configuration object
            
        Returns:
            Configured scheduler or None
        """
        scheduler_name = config.get("training.scheduler.name", None)
        
        if scheduler_name is None:
            return None
        
        if scheduler_name.lower() == "cosine":
            T_max = config.get("training.scheduler.T_max", 100)
            eta_min = config.get("training.scheduler.eta_min", 1e-6)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=T_max, eta_min=eta_min
            )
        
        elif scheduler_name.lower() == "cosine_warmup":
            scheduler = CosineAnnealingWarmupRestarts(
                optimizer,
                first_cycle_steps=config.get("training.scheduler.first_cycle_steps", 1000),
                cycle_mult=config.get("training.scheduler.cycle_mult", 1.0),
                max_lr=config.get("training.scheduler.max_lr", 1e-3),
                min_lr=config.get("training.scheduler.min_lr", 1e-6),
                warmup_steps=config.get("training.scheduler.warmup_steps", 100),
                gamma=config.get("training.scheduler.gamma", 1.0)
            )
        
        elif scheduler_name.lower() == "plateau":
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=config.get("training.scheduler.mode", "min"),
                factor=config.get("training.scheduler.factor", 0.5),
                patience=config.get("training.scheduler.patience", 5),
                min_lr=config.get("training.scheduler.min_lr", 1e-6)
            )
        
        elif scheduler_name.lower() == "step":
            scheduler = optim.lr_scheduler.StepLR(
                optimizer,
                step_size=config.get("training.scheduler.step_size", 30),
                gamma=config.get("training.scheduler.gamma", 0.1)
            )
        
        elif scheduler_name.lower() == "multistep":
            milestones = config.get("training.scheduler.milestones", [30, 60, 90])
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=milestones,
                gamma=config.get("training.scheduler.gamma", 0.1)
            )
        
        elif scheduler_name.lower() == "exponential":
            scheduler = optim.lr_scheduler.ExponentialLR(
                optimizer,
                gamma=config.get("training.scheduler.gamma", 0.95)
            )
        
        else:
            raise ValueError(f"Unknown scheduler: {scheduler_name}")
        
        logger.info(f"Created {scheduler_name} scheduler")
        return scheduler


class CosineAnnealingWarmupRestarts(_LRScheduler):
    """
    Cosine annealing with warmup and restarts.
    
    This scheduler implements cosine annealing with linear warmup and
    optional restarts, which is effective for training deep models.
    """
    
    def __init__(
        self,
        optimizer: optim.Optimizer,
        first_cycle_steps: int,
        cycle_mult: float = 1.0,
        max_lr: float = 0.1,
        min_lr: float = 0.001,
        warmup_steps: int = 0,
        gamma: float = 1.0,
        last_epoch: int = -1
    ):
        """
        Initialize cosine annealing with warmup restarts.
        
        Args:
            optimizer: Wrapped optimizer
            first_cycle_steps: Number of steps in first cycle
            cycle_mult: Cycle steps magnification
            max_lr: Maximum learning rate
            min_lr: Minimum learning rate
            warmup_steps: Number of warmup steps
            gamma: Decrease rate of max learning rate by cycle
            last_epoch: The index of last epoch
        """
        self.first_cycle_steps = first_cycle_steps
        self.cycle_mult = cycle_mult
        self.base_max_lr = max_lr
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.gamma = gamma
        
        self.cur_cycle_steps = first_cycle_steps
        self.cycle = 0
        self.step_in_cycle = last_epoch
        
        super().__init__(optimizer, last_epoch)
        
        # Initialize learning rates
        self.init_lr()
    
    def init_lr(self):
        """Initialize learning rates."""
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.min_lr
            self.base_lrs.append(self.min_lr)
    
    def get_lr(self):
        """Calculate learning rate."""
        if self.step_in_cycle == -1:
            return self.base_lrs
        elif self.step_in_cycle < self.warmup_steps:
            return [(self.max_lr - base_lr) * self.step_in_cycle / self.warmup_steps + base_lr
                    for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.max_lr - base_lr) *
                    (1 + math.cos(math.pi * (self.step_in_cycle - self.warmup_steps) /
                                  (self.cur_cycle_steps - self.warmup_steps))) / 2
                    for base_lr in self.base_lrs]
    
    def step(self, epoch=None):
        """Step the scheduler."""
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle = self.step_in_cycle + 1
            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle = self.step_in_cycle - self.cur_cycle_steps
                self.cur_cycle_steps = int((self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult) + self.warmup_steps
        else:
            if epoch >= self.first_cycle_steps:
                if self.cycle_mult == 1.0:
                    self.step_in_cycle = epoch % self.first_cycle_steps
                    self.cycle = epoch // self.first_cycle_steps
                else:
                    n = int(math.log((epoch / self.first_cycle_steps * (self.cycle_mult - 1) + 1), self.cycle_mult))
                    self.cycle = n
                    self.step_in_cycle = epoch - int(self.first_cycle_steps * (self.cycle_mult ** n - 1) / (self.cycle_mult - 1))
                    self.cur_cycle_steps = self.first_cycle_steps * self.cycle_mult ** n
            else:
                self.cur_cycle_steps = self.first_cycle_steps
                self.step_in_cycle = epoch
        
        self.max_lr = self.base_max_lr * (self.gamma ** self.cycle)
        self.last_epoch = math.floor(epoch)
        
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr
    
    def step_update(self, num_updates: int):
        """Update learning rate by step count."""
        self.step(num_updates)
