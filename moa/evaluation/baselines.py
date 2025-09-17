"""
Baseline models for MoA prediction comparison.

This module implements traditional machine learning baselines
for comprehensive comparison with the multi-modal deep learning approach.
"""

from typing import Dict, List, Tuple, Optional, Union, Any
from abc import ABC, abstractmethod
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import joblib

# Chemical fingerprints
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs
    from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    warnings.warn("RDKit not available. Chemical fingerprint baselines will be disabled.")

from moa.utils.config import Config
from moa.utils.logger import get_logger

logger = get_logger(__name__)


class BaselineModel(ABC):
    """Abstract base class for baseline models."""
    
    def __init__(self, name: str, config: Config):
        """
        Initialize baseline model.
        
        Args:
            name: Model name
            config: Configuration object
        """
        self.name = name
        self.config = config
        self.model = None
        self.scaler = None
        self.is_fitted = False
        
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the model to training data."""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on new data."""
        pass
    
    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        pass
    
    def save_model(self, filepath: str) -> None:
        """Save trained model to file."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'name': self.name,
            'config': self.config.to_dict()
        }
        joblib.dump(model_data, filepath)
        logger.info(f"Saved {self.name} model to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """Load trained model from file."""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.is_fitted = True
        logger.info(f"Loaded {self.name} model from {filepath}")


class ECFPRandomForestBaseline(BaselineModel):
    """ECFP fingerprints + Random Forest baseline."""
    
    def __init__(self, config: Config):
        """Initialize ECFP + Random Forest baseline."""
        super().__init__("ECFP_RandomForest", config)
        
        if not RDKIT_AVAILABLE:
            raise ImportError("RDKit is required for ECFP fingerprints")
        
        # ECFP parameters
        self.radius = config.get("baselines.ecfp.radius", 2)
        self.n_bits = config.get("baselines.ecfp.n_bits", 2048)
        
        # Random Forest parameters
        rf_params = config.get("baselines.random_forest", {})
        self.model = MultiOutputClassifier(
            RandomForestClassifier(
                n_estimators=rf_params.get("n_estimators", 100),
                max_depth=rf_params.get("max_depth", None),
                min_samples_split=rf_params.get("min_samples_split", 2),
                min_samples_leaf=rf_params.get("min_samples_leaf", 1),
                random_state=42,
                n_jobs=-1
            )
        )
        
        self.scaler = StandardScaler()
    
    def _smiles_to_ecfp(self, smiles_list: List[str]) -> np.ndarray:
        """Convert SMILES to ECFP fingerprints."""
        fingerprints = []
        
        for smiles in smiles_list:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    fp = GetMorganFingerprintAsBitVect(mol, self.radius, nBits=self.n_bits)
                    arr = np.zeros((self.n_bits,))
                    DataStructs.ConvertToNumpyArray(fp, arr)
                    fingerprints.append(arr)
                else:
                    # Invalid SMILES - use zero vector
                    fingerprints.append(np.zeros(self.n_bits))
            except:
                # Error processing SMILES - use zero vector
                fingerprints.append(np.zeros(self.n_bits))
        
        return np.array(fingerprints)
    
    def fit(self, smiles_list: List[str], y: np.ndarray) -> None:
        """Fit ECFP + Random Forest model."""
        logger.info(f"Computing ECFP fingerprints for {len(smiles_list)} compounds")
        X = self._smiles_to_ecfp(smiles_list)
        
        logger.info("Fitting Random Forest model")
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        
        logger.info(f"Fitted {self.name} model")
    
    def predict(self, smiles_list: List[str]) -> np.ndarray:
        """Make predictions using ECFP + Random Forest."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X = self._smiles_to_ecfp(smiles_list)
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, smiles_list: List[str]) -> np.ndarray:
        """Predict probabilities using ECFP + Random Forest."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X = self._smiles_to_ecfp(smiles_list)
        X_scaled = self.scaler.transform(X)
        
        # Get probabilities for each output
        probabilities = []
        for i, estimator in enumerate(self.model.estimators_):
            proba = estimator.predict_proba(X_scaled)
            # Take probability of positive class
            probabilities.append(proba[:, 1])
        
        return np.column_stack(probabilities)


class MorganSVMBaseline(BaselineModel):
    """Morgan fingerprints + SVM baseline."""
    
    def __init__(self, config: Config):
        """Initialize Morgan + SVM baseline."""
        super().__init__("Morgan_SVM", config)
        
        if not RDKIT_AVAILABLE:
            raise ImportError("RDKit is required for Morgan fingerprints")
        
        # Morgan parameters
        self.radius = config.get("baselines.morgan.radius", 2)
        self.n_bits = config.get("baselines.morgan.n_bits", 2048)
        
        # SVM parameters
        svm_params = config.get("baselines.svm", {})
        self.model = MultiOutputClassifier(
            SVC(
                C=svm_params.get("C", 1.0),
                kernel=svm_params.get("kernel", "rbf"),
                gamma=svm_params.get("gamma", "scale"),
                probability=True,  # Enable probability prediction
                random_state=42
            )
        )
        
        self.scaler = StandardScaler()
    
    def _smiles_to_morgan(self, smiles_list: List[str]) -> np.ndarray:
        """Convert SMILES to Morgan fingerprints."""
        fingerprints = []
        
        for smiles in smiles_list:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    fp = AllChem.GetMorganFingerprintAsBitVect(mol, self.radius, nBits=self.n_bits)
                    arr = np.zeros((self.n_bits,))
                    DataStructs.ConvertToNumpyArray(fp, arr)
                    fingerprints.append(arr)
                else:
                    fingerprints.append(np.zeros(self.n_bits))
            except:
                fingerprints.append(np.zeros(self.n_bits))
        
        return np.array(fingerprints)
    
    def fit(self, smiles_list: List[str], y: np.ndarray) -> None:
        """Fit Morgan + SVM model."""
        logger.info(f"Computing Morgan fingerprints for {len(smiles_list)} compounds")
        X = self._smiles_to_morgan(smiles_list)
        
        logger.info("Fitting SVM model")
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        
        logger.info(f"Fitted {self.name} model")
    
    def predict(self, smiles_list: List[str]) -> np.ndarray:
        """Make predictions using Morgan + SVM."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X = self._smiles_to_morgan(smiles_list)
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, smiles_list: List[str]) -> np.ndarray:
        """Predict probabilities using Morgan + SVM."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X = self._smiles_to_morgan(smiles_list)
        X_scaled = self.scaler.transform(X)
        
        # Get probabilities for each output
        probabilities = []
        for i, estimator in enumerate(self.model.estimators_):
            proba = estimator.predict_proba(X_scaled)
            probabilities.append(proba[:, 1])
        
        return np.column_stack(probabilities)


class LogisticRegressionBaseline(BaselineModel):
    """Logistic Regression baseline with chemical descriptors."""
    
    def __init__(self, config: Config):
        """Initialize Logistic Regression baseline."""
        super().__init__("LogisticRegression", config)
        
        # Logistic Regression parameters
        lr_params = config.get("baselines.logistic_regression", {})
        self.model = MultiOutputClassifier(
            LogisticRegression(
                C=lr_params.get("C", 1.0),
                penalty=lr_params.get("penalty", "l2"),
                solver=lr_params.get("solver", "liblinear"),
                max_iter=lr_params.get("max_iter", 1000),
                random_state=42
            )
        )
        
        self.scaler = StandardScaler()
    
    def _smiles_to_descriptors(self, smiles_list: List[str]) -> np.ndarray:
        """Convert SMILES to molecular descriptors."""
        if not RDKIT_AVAILABLE:
            raise ImportError("RDKit is required for molecular descriptors")
        
        from rdkit.Chem import Descriptors
        
        descriptors = []
        descriptor_names = [name for name, _ in Descriptors.descList]
        
        for smiles in smiles_list:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    desc_values = []
                    for desc_name, desc_func in Descriptors.descList:
                        try:
                            value = desc_func(mol)
                            # Handle NaN values
                            if np.isnan(value) or np.isinf(value):
                                value = 0.0
                            desc_values.append(value)
                        except:
                            desc_values.append(0.0)
                    descriptors.append(desc_values)
                else:
                    descriptors.append([0.0] * len(descriptor_names))
            except:
                descriptors.append([0.0] * len(descriptor_names))
        
        return np.array(descriptors)
    
    def fit(self, smiles_list: List[str], y: np.ndarray) -> None:
        """Fit Logistic Regression model."""
        logger.info(f"Computing molecular descriptors for {len(smiles_list)} compounds")
        X = self._smiles_to_descriptors(smiles_list)
        
        logger.info("Fitting Logistic Regression model")
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        
        logger.info(f"Fitted {self.name} model")
    
    def predict(self, smiles_list: List[str]) -> np.ndarray:
        """Make predictions using Logistic Regression."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X = self._smiles_to_descriptors(smiles_list)
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, smiles_list: List[str]) -> np.ndarray:
        """Predict probabilities using Logistic Regression."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X = self._smiles_to_descriptors(smiles_list)
        X_scaled = self.scaler.transform(X)
        
        # Get probabilities for each output
        probabilities = []
        for i, estimator in enumerate(self.model.estimators_):
            proba = estimator.predict_proba(X_scaled)
            probabilities.append(proba[:, 1])
        
        return np.column_stack(probabilities)


class GradientBoostingBaseline(BaselineModel):
    """Gradient Boosting baseline with ECFP fingerprints."""
    
    def __init__(self, config: Config):
        """Initialize Gradient Boosting baseline."""
        super().__init__("GradientBoosting", config)
        
        if not RDKIT_AVAILABLE:
            raise ImportError("RDKit is required for ECFP fingerprints")
        
        # ECFP parameters
        self.radius = config.get("baselines.ecfp.radius", 2)
        self.n_bits = config.get("baselines.ecfp.n_bits", 2048)
        
        # Gradient Boosting parameters
        gb_params = config.get("baselines.gradient_boosting", {})
        self.model = MultiOutputClassifier(
            GradientBoostingClassifier(
                n_estimators=gb_params.get("n_estimators", 100),
                learning_rate=gb_params.get("learning_rate", 0.1),
                max_depth=gb_params.get("max_depth", 3),
                random_state=42
            )
        )
        
        self.scaler = StandardScaler()
    
    def _smiles_to_ecfp(self, smiles_list: List[str]) -> np.ndarray:
        """Convert SMILES to ECFP fingerprints."""
        fingerprints = []
        
        for smiles in smiles_list:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    fp = GetMorganFingerprintAsBitVect(mol, self.radius, nBits=self.n_bits)
                    arr = np.zeros((self.n_bits,))
                    DataStructs.ConvertToNumpyArray(fp, arr)
                    fingerprints.append(arr)
                else:
                    fingerprints.append(np.zeros(self.n_bits))
            except:
                fingerprints.append(np.zeros(self.n_bits))
        
        return np.array(fingerprints)
    
    def fit(self, smiles_list: List[str], y: np.ndarray) -> None:
        """Fit Gradient Boosting model."""
        logger.info(f"Computing ECFP fingerprints for {len(smiles_list)} compounds")
        X = self._smiles_to_ecfp(smiles_list)
        
        logger.info("Fitting Gradient Boosting model")
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        
        logger.info(f"Fitted {self.name} model")
    
    def predict(self, smiles_list: List[str]) -> np.ndarray:
        """Make predictions using Gradient Boosting."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X = self._smiles_to_ecfp(smiles_list)
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, smiles_list: List[str]) -> np.ndarray:
        """Predict probabilities using Gradient Boosting."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X = self._smiles_to_ecfp(smiles_list)
        X_scaled = self.scaler.transform(X)
        
        # Get probabilities for each output
        probabilities = []
        for i, estimator in enumerate(self.model.estimators_):
            # For GradientBoostingClassifier, use decision_function and sigmoid
            decision = estimator.decision_function(X_scaled)
            proba = 1 / (1 + np.exp(-decision))  # Sigmoid
            probabilities.append(proba)
        
        return np.column_stack(probabilities)


class BaselineModels:
    """Factory class for creating and managing baseline models."""

    def __init__(self, config: Config):
        """
        Initialize baseline models factory.

        Args:
            config: Configuration object
        """
        self.config = config
        self.available_baselines = {
            'ecfp_rf': ECFPRandomForestBaseline,
            'morgan_svm': MorganSVMBaseline,
            'logistic_regression': LogisticRegressionBaseline,
            'gradient_boosting': GradientBoostingBaseline
        }

        # Check which baselines are available
        if not RDKIT_AVAILABLE:
            logger.warning("RDKit not available. Chemical fingerprint baselines disabled.")
            self.available_baselines = {
                'logistic_regression': LogisticRegressionBaseline
            }

    def create_baseline(self, baseline_name: str) -> BaselineModel:
        """
        Create a baseline model.

        Args:
            baseline_name: Name of baseline model

        Returns:
            Baseline model instance
        """
        if baseline_name not in self.available_baselines:
            available = list(self.available_baselines.keys())
            raise ValueError(f"Unknown baseline: {baseline_name}. Available: {available}")

        baseline_class = self.available_baselines[baseline_name]
        return baseline_class(self.config)

    def create_all_baselines(self) -> Dict[str, BaselineModel]:
        """
        Create all available baseline models.

        Returns:
            Dictionary of baseline models
        """
        baselines = {}
        for name in self.available_baselines:
            try:
                baselines[name] = self.create_baseline(name)
                logger.info(f"Created baseline: {name}")
            except Exception as e:
                logger.warning(f"Failed to create baseline {name}: {e}")

        return baselines

    def get_available_baselines(self) -> List[str]:
        """Get list of available baseline models."""
        return list(self.available_baselines.keys())

    def train_all_baselines(
        self,
        baselines: Dict[str, BaselineModel],
        smiles_list: List[str],
        targets: np.ndarray
    ) -> Dict[str, BaselineModel]:
        """
        Train all baseline models.

        Args:
            baselines: Dictionary of baseline models
            smiles_list: List of SMILES strings
            targets: Target labels

        Returns:
            Dictionary of trained baseline models
        """
        trained_baselines = {}

        for name, model in baselines.items():
            try:
                logger.info(f"Training baseline: {name}")
                model.fit(smiles_list, targets)
                trained_baselines[name] = model
                logger.info(f"Successfully trained {name}")
            except Exception as e:
                logger.error(f"Failed to train baseline {name}: {e}")

        return trained_baselines

    def evaluate_baselines(
        self,
        baselines: Dict[str, BaselineModel],
        smiles_list: List[str],
        targets: np.ndarray,
        metrics_calculator
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate all baseline models.

        Args:
            baselines: Dictionary of trained baseline models
            smiles_list: List of SMILES strings for evaluation
            targets: True target labels
            metrics_calculator: Metrics calculator instance

        Returns:
            Dictionary of evaluation results
        """
        results = {}

        for name, model in baselines.items():
            try:
                logger.info(f"Evaluating baseline: {name}")

                # Make predictions
                predictions = model.predict(smiles_list)
                probabilities = model.predict_proba(smiles_list)

                # Compute metrics
                metrics = metrics_calculator.compute_moa_specific_metrics(
                    targets, predictions, probabilities
                )

                results[name] = metrics
                logger.info(f"Evaluated {name} - AUROC: {metrics.get('auroc_macro', 0):.4f}")

            except Exception as e:
                logger.error(f"Failed to evaluate baseline {name}: {e}")
                results[name] = {}

        return results

    def save_baselines(
        self,
        baselines: Dict[str, BaselineModel],
        save_dir: str
    ) -> None:
        """
        Save all trained baseline models.

        Args:
            baselines: Dictionary of trained baseline models
            save_dir: Directory to save models
        """
        import os
        os.makedirs(save_dir, exist_ok=True)

        for name, model in baselines.items():
            try:
                filepath = os.path.join(save_dir, f"{name}_model.pkl")
                model.save_model(filepath)
            except Exception as e:
                logger.error(f"Failed to save baseline {name}: {e}")

    def load_baselines(
        self,
        baseline_names: List[str],
        save_dir: str
    ) -> Dict[str, BaselineModel]:
        """
        Load trained baseline models.

        Args:
            baseline_names: List of baseline names to load
            save_dir: Directory containing saved models

        Returns:
            Dictionary of loaded baseline models
        """
        import os
        baselines = {}

        for name in baseline_names:
            try:
                filepath = os.path.join(save_dir, f"{name}_model.pkl")
                if os.path.exists(filepath):
                    model = self.create_baseline(name)
                    model.load_model(filepath)
                    baselines[name] = model
                    logger.info(f"Loaded baseline: {name}")
                else:
                    logger.warning(f"Model file not found: {filepath}")
            except Exception as e:
                logger.error(f"Failed to load baseline {name}: {e}")

        return baselines
