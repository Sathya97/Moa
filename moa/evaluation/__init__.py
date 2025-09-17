"""
Evaluation framework for MoA prediction.

This module contains comprehensive evaluation tools including:
- Multi-label classification metrics
- Statistical significance testing
- Baseline model comparisons
- Performance visualization
- Model interpretation analysis
"""

from .metrics import MoAMetrics, MultiLabelMetrics
from .evaluator import MoAEvaluator
from .baselines import BaselineModels
from .statistical_tests import StatisticalTests
from .visualization import EvaluationVisualizer
from .interpretation import ModelInterpreter

__all__ = [
    'MoAMetrics',
    'MultiLabelMetrics',
    'MoAEvaluator',
    'BaselineModels',
    'StatisticalTests',
    'EvaluationVisualizer',
    'ModelInterpreter'
]
