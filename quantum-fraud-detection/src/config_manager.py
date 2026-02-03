"""
Configuration Management System
Handles loading, validation, and merging of configuration from multiple sources.
"""
from __future__ import annotations
import os
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from dotenv import load_dotenv
import hashlib
import json

logger = logging.getLogger(__name__)


class ConfigValidationError(Exception):
    """Raised when configuration validation fails."""
    pass


@dataclass
class ExperimentConfig:
    """Experiment metadata and tracking."""
    name: str
    version: str
    description: str
    random_seed: int = 42
    log_level: str = "INFO"
    
    def __post_init__(self):
        """Validate experiment configuration."""
        if not self.name:
            raise ConfigValidationError("Experiment name cannot be empty")
        if self.random_seed < 0:
            raise ConfigValidationError("Random seed must be non-negative")


@dataclass
class DataConfig:
    """Data loading and splitting configuration."""
    transaction_csv: str
    identity_csv: str
    target_col: str = "isFraud"
    id_cols: list = field(default_factory=lambda: ["TransactionID"])
    nrows: Optional[int] = None
    use_time_based_split: bool = True
    time_col: str = "TransactionDT"
    test_size: float = 0.2
    
    def __post_init__(self):
        """Validate data configuration."""
        if not os.path.exists(self.transaction_csv) and self.nrows is None:
            logger.warning(f"Transaction file not found: {self.transaction_csv}")
        if self.test_size <= 0 or self.test_size >= 1:
            raise ConfigValidationError("test_size must be between 0 and 1")


@dataclass
class PreprocessingConfig:
    """Preprocessing pipeline configuration."""
    feature_engineering: Dict[str, bool]
    missing_threshold: float = 50.0
    imputation_strategy: str = "median"
    feature_selection: Dict[str, Any] = field(default_factory=dict)
    scaler: str = "standard"
    resampling: Dict[str, Any] = field(default_factory=dict)
    llm_features: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate preprocessing configuration."""
        if self.missing_threshold < 0 or self.missing_threshold > 100:
            raise ConfigValidationError("missing_threshold must be between 0 and 100")
        if self.imputation_strategy not in ["median", "mean", "mode", "constant"]:
            raise ConfigValidationError(f"Invalid imputation strategy: {self.imputation_strategy}")
        if self.scaler not in ["standard", "minmax", "robust"]:
            raise ConfigValidationError(f"Invalid scaler: {self.scaler}")


@dataclass
class QuantumBackendConfig:
    """Quantum backend configuration."""
    type: str = "simulator"
    ibm_token: Optional[str] = None
    ibm_backend_name: str = "ibm_torino"
    ibm_instance: Optional[str] = None
    shots: int = 1024
    optimization_level: int = 3
    resilience_level: int = 1
    max_circuits_per_job: int = 100
    max_shots_per_circuit: int = 8192
    job_timeout: int = 3600
    
    def __post_init__(self):
        """Validate quantum backend configuration."""
        if self.type not in ["simulator", "aer", "ibm_quantum"]:
            raise ConfigValidationError(f"Invalid backend type: {self.type}")
        
        if self.type == "ibm_quantum" and not self.ibm_token:
            # Try to load from environment
            self.ibm_token = os.getenv("IBM_QUANTUM_TOKEN")
            if not self.ibm_token:
                raise ConfigValidationError(
                    "IBM Quantum backend requires ibm_token. "
                    "Set it in config or IBM_QUANTUM_TOKEN environment variable."
                )
        
        if self.shots < 1 or self.shots > self.max_shots_per_circuit:
            raise ConfigValidationError(f"shots must be between 1 and {self.max_shots_per_circuit}")
        
        if self.optimization_level not in [0, 1, 2, 3]:
            raise ConfigValidationError("optimization_level must be 0, 1, 2, or 3")


class ConfigManager:
    """
    Central configuration management system.
    Handles loading, validation, environment variable substitution, and config merging.
    """
    
    def __init__(self, config_path: str, load_env: bool = True):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to main YAML configuration file
            load_env: Whether to load .env file
        """
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        # Load environment variables
        if load_env:
            env_file = self.config_path.parent.parent / ".env"
            if env_file.exists():
                load_dotenv(env_file)
                logger.info(f"Loaded environment from {env_file}")
        
        # Load and validate configuration
        self.raw_config = self._load_yaml(self.config_path)
        self.config = self._substitute_env_vars(self.raw_config)
        self._validate_config()
        
        # Generate unique experiment ID
        self.experiment_id = self._generate_experiment_id()
        logger.info(f"Configuration loaded successfully. Experiment ID: {self.experiment_id}")
    
    def _load_yaml(self, path: Path) -> Dict[str, Any]:
        """Load YAML configuration file."""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            return config
        except yaml.YAMLError as e:
            raise ConfigValidationError(f"Invalid YAML syntax in {path}: {e}")
    
    def _substitute_env_vars(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recursively substitute environment variables in configuration.
        Syntax: ${ENV_VAR_NAME} or ${ENV_VAR_NAME:default_value}
        """
        if isinstance(config, dict):
            return {k: self._substitute_env_vars(v) for k, v in config.items()}
        elif isinstance(config, list):
            return [self._substitute_env_vars(item) for item in config]
        elif isinstance(config, str):
            # Check for environment variable syntax
            if config.startswith("${") and config.endswith("}"):
                env_expr = config[2:-1]  # Remove ${ and }
                
                # Check for default value
                if ":" in env_expr:
                    var_name, default = env_expr.split(":", 1)
                    return os.getenv(var_name, default)
                else:
                    value = os.getenv(env_expr)
                    if value is None:
                        raise ConfigValidationError(
                            f"Environment variable {env_expr} not set and no default provided"
                        )
                    return value
            return config
        else:
            return config
    
    def _validate_config(self):
        """Validate configuration structure and values."""
        required_sections = ["experiment", "data", "preprocessing", "paths"]
        for section in required_sections:
            if section not in self.config:
                raise ConfigValidationError(f"Missing required section: {section}")
        
        # Validate experiment config
        try:
            ExperimentConfig(**self.config["experiment"])
        except TypeError as e:
            raise ConfigValidationError(f"Invalid experiment configuration: {e}")
        
        # Validate data config
        try:
            DataConfig(**self.config["data"])
        except TypeError as e:
            raise ConfigValidationError(f"Invalid data configuration: {e}")
        
        # Validate quantum backend if quantum models enabled
        if self.config.get("quantum_models", {}).get("enabled", False):
            backend_config = self.config["quantum_models"]["backend"]
            try:
                QuantumBackendConfig(**backend_config)
            except TypeError as e:
                raise ConfigValidationError(f"Invalid quantum backend configuration: {e}")
    
    def _generate_experiment_id(self) -> str:
        """Generate unique experiment ID based on configuration."""
        # Create deterministic hash of configuration
        config_str = json.dumps(self.config, sort_keys=True)
        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
        
        exp_name = self.config["experiment"]["name"]
        exp_version = self.config["experiment"]["version"]
        
        return f"{exp_name}_v{exp_version}_{config_hash}"
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Example:
            config.get("quantum_models.vqc.optimizer.maxiter")
        """
        keys = key_path.split(".")
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """Get entire configuration section."""
        return self.config.get(section, {})
    
    def set_global_seed(self):
        """Set global random seed for reproducibility."""
        import random
        import numpy as np
        
        seed = self.config["experiment"]["random_seed"]
        random.seed(seed)
        np.random.seed(seed)
        
        # Try to set torch seed if available
        try:
            import torch
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except ImportError:
            pass
        
        logger.info(f"Global random seed set to {seed}")
    
    def setup_logging(self):
        """Configure logging based on config."""
        log_level = self.config["experiment"]["log_level"]
        logging.basicConfig(
            level=getattr(logging, log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(
                    Path(self.config["paths"]["logs_dir"]) / f"{self.experiment_id}.log"
                )
            ]
        )
    
    def create_output_directories(self):
        """Create all output directories specified in config."""
        paths_config = self.config["paths"]
        
        for key, path in paths_config.items():
            if key.endswith("_dir"):
                Path(path).mkdir(parents=True, exist_ok=True)
                logger.debug(f"Created directory: {path}")
    
    def save_config_snapshot(self, output_path: Optional[Path] = None):
        """Save configuration snapshot for reproducibility."""
        if output_path is None:
            output_path = Path(self.config["paths"]["artifacts_dir"]) / f"{self.experiment_id}_config.yaml"
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)
        
        logger.info(f"Configuration snapshot saved to {output_path}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Return full configuration as dictionary."""
        return self.config.copy()


def load_config(config_path: str = "configs/config_master.yaml", **overrides) -> ConfigManager:
    """
    Convenience function to load configuration.
    
    Args:
        config_path: Path to configuration file
        **overrides: Configuration overrides (e.g., random_seed=123)
    
    Returns:
        ConfigManager instance
    """
    manager = ConfigManager(config_path)
    
    # Apply overrides
    for key, value in overrides.items():
        keys = key.split(".")
        config_dict = manager.config
        
        for k in keys[:-1]:
            config_dict = config_dict.setdefault(k, {})
        config_dict[keys[-1]] = value
    
    return manager


if __name__ == "__main__":
    # Test configuration loading
    print("Testing configuration system...")
    
    try:
        config = load_config()
        print(f"✓ Configuration loaded successfully")
        print(f"✓ Experiment ID: {config.experiment_id}")
        print(f"✓ Random seed: {config.get('experiment.random_seed')}")
        print(f"✓ Backend type: {config.get('quantum_models.backend.type')}")
    except Exception as e:
        print(f"✗ Configuration loading failed: {e}")
