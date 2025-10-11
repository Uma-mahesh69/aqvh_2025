from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Literal
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    # optional, used if installed
    from imblearn.over_sampling import RandomOverSampler
    from imblearn.pipeline import Pipeline as ImbPipeline
    IMB_AVAILABLE = True
except Exception:
    IMB_AVAILABLE = False


@dataclass
class ClassicalConfig:
    penalty: str = "l2"
    C: float = 1.0
    max_iter: int = 1000
    class_weight: Optional[str] = None  # e.g., "balanced"
    use_random_oversampler: bool = True


def build_logreg(cfg: ClassicalConfig) -> Pipeline:
    """Build logistic regression pipeline with optional oversampling.
    
    Args:
        cfg: Classical model configuration
        
    Returns:
        Sklearn pipeline with scaler, optional sampler, and classifier
    """
    lr = LogisticRegression(
        penalty=cfg.penalty,
        C=cfg.C,
        max_iter=cfg.max_iter,
        class_weight=cfg.class_weight,
        solver="lbfgs",
        n_jobs=None,
    )
    steps = [("scaler", StandardScaler(with_mean=False)), ("clf", lr)]

    if cfg.use_random_oversampler and IMB_AVAILABLE:
        # Oversample only during fit
        sampler = RandomOverSampler()
        pipe: Pipeline = ImbPipeline([("sampler", sampler)] + steps)
        return pipe

    return Pipeline(steps)


def train_logreg(X_train: np.ndarray, y_train: np.ndarray, cfg: ClassicalConfig) -> Pipeline:
    """Train logistic regression model with optional oversampling.
    
    Args:
        X_train: Training features
        y_train: Training labels
        cfg: Classical model configuration
        
    Returns:
        Trained pipeline
    """
    pipe = build_logreg(cfg)
    pipe.fit(X_train, y_train)
    return pipe


@dataclass
class IsolationForestConfig:
    n_estimators: int = 100
    contamination: float = 0.1
    max_samples: str = "auto"
    random_state: int = 42


def build_isolation_forest(cfg: IsolationForestConfig) -> IsolationForest:
    """Build Isolation Forest model for anomaly detection.
    
    Args:
        cfg: Isolation Forest configuration
        
    Returns:
        Sklearn IsolationForest model
    """
    model = IsolationForest(
        n_estimators=cfg.n_estimators,
        contamination=cfg.contamination,
        max_samples=cfg.max_samples,
        random_state=cfg.random_state,
        n_jobs=-1,
    )
    return model


def train_isolation_forest(X_train: np.ndarray, y_train: np.ndarray, cfg: IsolationForestConfig) -> IsolationForest:
    """Train Isolation Forest model.
    
    Args:
        X_train: Training features
        y_train: Training labels (used to set contamination if needed)
        cfg: Isolation Forest configuration
        
    Returns:
        Trained Isolation Forest model
    """
    model = build_isolation_forest(cfg)
    model.fit(X_train)
    return model


@dataclass
class XGBoostConfig:
    n_estimators: int = 100
    max_depth: int = 6
    learning_rate: float = 0.1
    scale_pos_weight: Optional[float] = None
    random_state: int = 42
    use_gpu: bool = False


def build_xgboost(cfg: XGBoostConfig):
    """Build XGBoost classifier.
    
    Args:
        cfg: XGBoost configuration
        
    Returns:
        XGBoost classifier
        
    Raises:
        ImportError: If XGBoost is not installed
    """
    if not XGB_AVAILABLE:
        raise ImportError("XGBoost is not installed. Install with: pip install xgboost")
    
    params = {
        "n_estimators": cfg.n_estimators,
        "max_depth": cfg.max_depth,
        "learning_rate": cfg.learning_rate,
        "random_state": cfg.random_state,
        "eval_metric": "logloss",
        "use_label_encoder": False,
    }
    
    if cfg.scale_pos_weight is not None:
        params["scale_pos_weight"] = cfg.scale_pos_weight
    
    if cfg.use_gpu:
        params["tree_method"] = "gpu_hist"
        params["predictor"] = "gpu_predictor"
    
    model = xgb.XGBClassifier(**params)
    return model


def train_xgboost(X_train: np.ndarray, y_train: np.ndarray, cfg: XGBoostConfig):
    """Train XGBoost classifier.
    
    Args:
        X_train: Training features
        y_train: Training labels
        cfg: XGBoost configuration
        
    Returns:
        Trained XGBoost model
    """
    model = build_xgboost(cfg)
    model.fit(X_train, y_train)
    return model
