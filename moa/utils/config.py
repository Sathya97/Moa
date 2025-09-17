"""Configuration management utilities."""

import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml
from omegaconf import DictConfig, OmegaConf


class Config:
    """Configuration manager for the MoA prediction framework."""
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize configuration.
        
        Args:
            config_path: Path to configuration file. If None, uses default config.
        """
        if config_path is None:
            # Use default config from project root
            project_root = Path(__file__).parent.parent.parent
            config_path = project_root / "configs" / "config.yaml"
        
        self.config_path = Path(config_path)
        self._config = self._load_config()
        
    def _load_config(self) -> DictConfig:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return OmegaConf.create(config_dict)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key.

        Args:
            key: Configuration key (supports dot notation, e.g., 'model.hidden_dim')
            default: Default value if key not found

        Returns:
            Configuration value
        """
        try:
            value = OmegaConf.select(self._config, key)
            # Return default if value is None (key doesn't exist)
            return value if value is not None else default
        except Exception:
            return default
    
    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value.
        
        Args:
            key: Configuration key (supports dot notation)
            value: Value to set
        """
        OmegaConf.set(self._config, key, value)
    
    def update(self, updates: Dict[str, Any]) -> None:
        """
        Update configuration with dictionary of values.
        
        Args:
            updates: Dictionary of key-value pairs to update
        """
        self._config = OmegaConf.merge(self._config, updates)
    
    def save(self, path: Optional[Union[str, Path]] = None) -> None:
        """
        Save configuration to file.
        
        Args:
            path: Path to save configuration. If None, overwrites original file.
        """
        save_path = Path(path) if path else self.config_path
        
        with open(save_path, 'w') as f:
            yaml.dump(OmegaConf.to_yaml(self._config), f, default_flow_style=False)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return OmegaConf.to_container(self._config, resolve=True)
    
    @property
    def config(self) -> DictConfig:
        """Get the underlying OmegaConf configuration."""
        return self._config
    
    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style access."""
        return self.get(key)
    
    def __setitem__(self, key: str, value: Any) -> None:
        """Allow dictionary-style setting."""
        self.set(key, value)
    
    def __contains__(self, key: str) -> bool:
        """Check if key exists in configuration."""
        return OmegaConf.select(self._config, key) is not None


def load_config(config_path: Optional[Union[str, Path]] = None) -> Config:
    """
    Load configuration from file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Config object
    """
    return Config(config_path)


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent.parent


def setup_paths(config: Config) -> None:
    """
    Setup and create necessary directories based on configuration.
    
    Args:
        config: Configuration object
    """
    project_root = get_project_root()
    
    # Create data directories
    data_dir = project_root / config.get("paths.data_dir", "data")
    (data_dir / "raw").mkdir(parents=True, exist_ok=True)
    (data_dir / "processed").mkdir(parents=True, exist_ok=True)
    (data_dir / "splits").mkdir(parents=True, exist_ok=True)
    
    # Create model and results directories
    (project_root / config.get("paths.model_dir", "models")).mkdir(parents=True, exist_ok=True)
    (project_root / config.get("paths.results_dir", "results")).mkdir(parents=True, exist_ok=True)
    (project_root / config.get("paths.cache_dir", "cache")).mkdir(parents=True, exist_ok=True)
