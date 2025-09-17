"""
Mechanism of Action (MoA) Prediction Framework

A comprehensive framework for predicting drug mechanisms of action using
multi-modal deep learning approaches combining chemical, biological, and
structural information.
"""

__version__ = "0.1.0"
__author__ = "MoA Research Team"
__email__ = "research@moa-prediction.org"

from moa.utils.config import Config
from moa.utils.logger import get_logger

__all__ = ["Config", "get_logger"]
