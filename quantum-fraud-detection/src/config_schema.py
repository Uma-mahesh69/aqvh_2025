from pydantic import BaseModel, Field, validator
from typing import List, Optional, Literal, Dict

class DataConfig(BaseModel):
    transaction_csv: str
    identity_csv: str
    nrows: Optional[int] = Field(None, gt=0, description="Number of rows to load. None for all.")
    target_col: str = "isFraud"

class PreprocessingConfig(BaseModel):
    missing_threshold: float = Field(50.0, ge=0.0, le=100.0)
    top_k_features: int = Field(8, gt=0)
    feature_selection_method: str = "pca"
    llm_n_components: int = Field(3, gt=0)
    use_smote: bool = False
    smote_k_neighbors: int = 5
    test_size: float = Field(0.2, gt=0.0, lt=1.0)
    random_state: int = 42
    stratify: bool = True
    
    class Config:
        extra = "allow" 

class ModelsToRunConfig(BaseModel):
    logistic_regression: bool = True
    isolation_forest: bool = True
    xgboost: bool = True
    quantum_vqc: bool = False
    quantum_kernel: bool = False

class LogisticRegressionConfig(BaseModel):
    max_iter: int = Field(1000, gt=0)
    C: float = Field(1.0, gt=0.0)
    penalty: Literal["l1", "l2", "elasticnet", "none"] = "l2"
    class_weight: Optional[str] = None
    use_random_oversampler: bool = True

class IsolationForestConfig(BaseModel):
    n_estimators: int = 100
    contamination: float = 0.1
    max_samples: str = "auto"
    random_state: int = 42

class XGBoostConfig(BaseModel):
    n_estimators: int = 100
    learning_rate: float = 0.1
    max_depth: int = 3
    scale_pos_weight: Optional[float] = 1.0
    subsample: float = 1.0
    colsample_bytree: float = 1.0
    random_state: int = 42
    use_gpu: bool = False

class QuantumVQCConfig(BaseModel):
    reps_feature_map: int = 1
    reps_ansatz: int = 1
    optimizer_maxiter: int = 50
    shots: Optional[int] = 1024
    entanglement: str = "linear"
    optimizer: str = "cobyla"

class QuantumKernelConfig(BaseModel):
    reps_feature_map: int = 1
    shots: Optional[int] = 1024
    C: float = 1.0
    gamma: str = "scale"

class QuantumBackendConfig(BaseModel):
    backend_type: Literal["simulator", "aer", "ibm_quantum"]
    ibm_token: Optional[str] = None
    ibm_backend_name: Optional[str] = "ibmq_qasm_simulator"
    shots: int = 1024
    optimization_level: int = Field(3, ge=0, le=3)
    resilience_level: int = Field(1, ge=0, le=3)

class PathConfig(BaseModel):
    results_dir: str = "results"
    logs_dir: str = "results/logs"
    figures_dir: str = "results/figures"

class AppConfig(BaseModel):
    data: DataConfig
    preprocessing: PreprocessingConfig
    models_to_run: ModelsToRunConfig = Field(default_factory=ModelsToRunConfig)
    logistic_regression: LogisticRegressionConfig = Field(default_factory=LogisticRegressionConfig)
    isolation_forest: IsolationForestConfig = Field(default_factory=IsolationForestConfig)
    xgboost: XGBoostConfig = Field(default_factory=XGBoostConfig)
    quantum_vqc: QuantumVQCConfig = Field(default_factory=QuantumVQCConfig)
    quantum_kernel: QuantumKernelConfig = Field(default_factory=QuantumKernelConfig)
    quantum_backend: QuantumBackendConfig = Field(default_factory=lambda: QuantumBackendConfig(backend_type="simulator"))
    paths: PathConfig = Field(default_factory=PathConfig)

def validate_config(config_dict: Dict) -> AppConfig:
    """Validates dictionary against AppConfig schema."""
    return AppConfig(**config_dict)
