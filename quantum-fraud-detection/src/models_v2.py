"""
Unified Model Training System
Handles classical and quantum models with consistent interface.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import logging
import time
from typing import Dict, Any, Tuple, Optional, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod

# Classical ML
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as xgb

# Quantum ML
from qiskit import QuantumCircuit
from qiskit.circuit.library import ZZFeatureMap, ZFeatureMap, RealAmplitudes, EfficientSU2, TwoLocal
from qiskit.primitives import Estimator as LocalEstimator, Sampler as LocalSampler
from qiskit_aer.primitives import Estimator as AerEstimator, Sampler as AerSampler
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_algorithms.optimizers import COBYLA, SPSA, ADAM
from qiskit_algorithms.state_fidelities import ComputeUncompute

# IBM Quantum Runtime
try:
    from qiskit_ibm_runtime import QiskitRuntimeService, Estimator as IBMEstimator, SamplerV2 as IBMSampler
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    IBM_RUNTIME_AVAILABLE = True
except ImportError:
    IBM_RUNTIME_AVAILABLE = False

logger = logging.getLogger(__name__)


# ==============================================================================
# BASE MODEL INTERFACE
# ==============================================================================

class BaseModel(ABC):
    """Abstract base class for all models."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.training_time = 0.0
        self.is_fitted = False
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BaseModel':
        """Train the model."""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        pass
    
    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities."""
        pass
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        return self.config


# ==============================================================================
# CLASSICAL MODELS
# ==============================================================================

class LogisticRegressionModel(BaseModel):
    """Logistic Regression with proper configuration."""
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LogisticRegressionModel':
        """Train logistic regression."""
        logger.info("Training Logistic Regression...")
        start_time = time.time()
        
        self.model = LogisticRegression(
            max_iter=self.config.get('max_iter', 1000),
            C=self.config.get('C', 1.0),
            penalty=self.config.get('penalty', 'l2'),
            solver=self.config.get('solver', 'lbfgs'),
            class_weight=self.config.get('class_weight', 'balanced'),
            n_jobs=self.config.get('n_jobs', -1),
            random_state=self.config.get('random_state', 42)
        )
        
        self.model.fit(X, y)
        self.training_time = time.time() - start_time
        self.is_fitted = True
        
        logger.info(f"Logistic Regression trained in {self.training_time:.2f}s")
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        return self.model.predict_proba(X)


class RandomForestModel(BaseModel):
    """Random Forest with proper configuration."""
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RandomForestModel':
        """Train random forest."""
        logger.info("Training Random Forest...")
        start_time = time.time()
        
        self.model = RandomForestClassifier(
            n_estimators=self.config.get('n_estimators', 100),
            max_depth=self.config.get('max_depth', 10),
            min_samples_split=self.config.get('min_samples_split', 10),
            min_samples_leaf=self.config.get('min_samples_leaf', 4),
            class_weight=self.config.get('class_weight', 'balanced'),
            n_jobs=self.config.get('n_jobs', -1),
            random_state=self.config.get('random_state', 42)
        )
        
        self.model.fit(X, y)
        self.training_time = time.time() - start_time
        self.is_fitted = True
        
        logger.info(f"Random Forest trained in {self.training_time:.2f}s")
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        return self.model.predict_proba(X)


class XGBoostModel(BaseModel):
    """XGBoost with proper configuration and early stopping."""
    
    def fit(self, X: np.ndarray, y: np.ndarray, X_val: Optional[np.ndarray] = None, 
            y_val: Optional[np.ndarray] = None) -> 'XGBoostModel':
        """Train XGBoost with optional validation set."""
        logger.info("Training XGBoost...")
        start_time = time.time()
        
        # Auto-calculate scale_pos_weight if not provided
        scale_pos_weight = self.config.get('scale_pos_weight')
        if scale_pos_weight is None:
            n_negative = (y == 0).sum()
            n_positive = (y == 1).sum()
            scale_pos_weight = n_negative / n_positive if n_positive > 0 else 1.0
            logger.info(f"Auto-calculated scale_pos_weight: {scale_pos_weight:.2f}")
        
        self.model = xgb.XGBClassifier(
            n_estimators=self.config.get('n_estimators', 200),
            max_depth=self.config.get('max_depth', 8),
            learning_rate=self.config.get('learning_rate', 0.05),
            subsample=self.config.get('subsample', 0.8),
            colsample_bytree=self.config.get('colsample_bytree', 0.4),
            scale_pos_weight=scale_pos_weight,
            gamma=self.config.get('gamma', 0.1),
            reg_alpha=self.config.get('reg_alpha', 0.1),
            reg_lambda=self.config.get('reg_lambda', 1.0),
            eval_metric=self.config.get('eval_metric', 'logloss'),
            n_jobs=self.config.get('n_jobs', -1),
            random_state=self.config.get('random_state', 42),
            use_label_encoder=False
        )
        
        # Train with early stopping if validation set provided
        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]
            early_stopping_rounds = self.config.get('early_stopping_rounds', 50)
            
            self.model.fit(
                X, y,
                eval_set=eval_set,
                verbose=False
            )
        else:
            self.model.fit(X, y)
        
        self.training_time = time.time() - start_time
        self.is_fitted = True
        
        logger.info(f"XGBoost trained in {self.training_time:.2f}s")
        if hasattr(self.model, 'best_iteration'):
            logger.info(f"Best iteration: {self.model.best_iteration}")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        return self.model.predict_proba(X)


# ==============================================================================
# QUANTUM MODELS
# ==============================================================================

class QuantumBackend:
    """Handles quantum backend configuration and primitive creation."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.backend_type = config.get('type', 'simulator')
        self.backend = None
        self._validate_config()
    
    def _validate_config(self):
        """Validate quantum backend configuration."""
        if self.backend_type == 'ibm_quantum' and not IBM_RUNTIME_AVAILABLE:
            raise ImportError(
                "qiskit-ibm-runtime not installed. "
                "Install with: pip install qiskit-ibm-runtime"
            )
        
        if self.backend_type == 'ibm_quantum':
            token = self.config.get('ibm_token') or os.getenv('IBM_QUANTUM_TOKEN')
            if not token:
                raise ValueError(
                    "IBM Quantum token required. Set in config or IBM_QUANTUM_TOKEN env var"
                )
    
    def get_estimator(self):
        """Get appropriate Estimator primitive."""
        if self.backend_type == 'simulator':
            return LocalEstimator()
        
        elif self.backend_type == 'aer':
            return AerEstimator()
        
        elif self.backend_type == 'ibm_quantum':
            # Initialize IBM Quantum service
            token = self.config.get('ibm_token') or os.getenv('IBM_QUANTUM_TOKEN')
            service = QiskitRuntimeService(channel='ibm_quantum', token=token)
            
            backend_name = self.config.get('ibm_backend_name', 'ibm_torino')
            backend = service.backend(backend_name)
            self.backend = backend
            
            logger.info(f"Using IBM Quantum backend: {backend_name}")
            
            # Create Estimator with error mitigation
            estimator = IBMEstimator(mode=backend)
            estimator.options.resilience_level = self.config.get('resilience_level', 1)
            estimator.options.optimization_level = self.config.get('optimization_level', 3)
            
            if self.config.get('shots'):
                estimator.options.default_shots = self.config['shots']
            
            return estimator
        
        else:
            raise ValueError(f"Unknown backend type: {self.backend_type}")
    
    def get_sampler(self):
        """Get appropriate Sampler primitive."""
        if self.backend_type == 'simulator':
            try:
                from qiskit.primitives import StatevectorSampler
                return StatevectorSampler()
            except ImportError:
                return LocalSampler()
        
        elif self.backend_type == 'aer':
            return AerSampler()
        
        elif self.backend_type == 'ibm_quantum':
            token = self.config.get('ibm_token') or os.getenv('IBM_QUANTUM_TOKEN')
            service = QiskitRuntimeService(channel='ibm_quantum', token=token)
            
            backend_name = self.config.get('ibm_backend_name', 'ibm_torino')
            backend = service.backend(backend_name)
            self.backend = backend
            
            sampler = IBMSampler(mode=backend)
            if self.config.get('shots'):
                sampler.options.default_shots = self.config['shots']
            
            return sampler
        
        else:
            raise ValueError(f"Unknown backend type: {self.backend_type}")
    
    def transpile_circuit(self, circuit: QuantumCircuit) -> QuantumCircuit:
        """Transpile circuit for backend if needed."""
        if self.backend_type != 'ibm_quantum' or self.backend is None:
            return circuit
        
        logger.info("Transpiling circuit for IBM Quantum backend...")
        
        # Use preset pass manager for optimization
        pm = generate_preset_pass_manager(
            optimization_level=self.config.get('optimization_level', 3),
            target=self.backend.target
        )
        
        transpiled = pm.run(circuit)
        logger.info(f"Circuit transpiled: {transpiled.num_qubits} qubits, "
                   f"{transpiled.depth()} depth, {transpiled.size()} gates")
        
        return transpiled


class QuantumVQCModel(BaseModel):
    """Variational Quantum Classifier."""
    
    def __init__(self, config: Dict[str, Any], n_features: int):
        super().__init__(config)
        self.n_features = n_features
        self.backend_config = config.get('backend', {})
        self.vqc_config = config.get('vqc', {})
        self.quantum_backend = QuantumBackend(self.backend_config)
        self._build_circuit()
    
    def _build_circuit(self):
        """Build quantum circuit for VQC."""
        # Feature map
        fm_config = self.vqc_config.get('feature_map', {})
        fm_type = fm_config.get('type', 'ZZFeatureMap')
        fm_reps = fm_config.get('reps', 2)
        entanglement = fm_config.get('entanglement', 'linear')
        
        if fm_type == 'ZZFeatureMap':
            self.feature_map = ZZFeatureMap(
                feature_dimension=self.n_features,
                reps=fm_reps,
                entanglement=entanglement
            )
        elif fm_type == 'ZFeatureMap':
            self.feature_map = ZFeatureMap(
                feature_dimension=self.n_features,
                reps=fm_reps
            )
        else:
            raise ValueError(f"Unknown feature map type: {fm_type}")
        
        # Ansatz
        ansatz_config = self.vqc_config.get('ansatz', {})
        ansatz_type = ansatz_config.get('type', 'RealAmplitudes')
        ansatz_reps = ansatz_config.get('reps', 2)
        
        if ansatz_type == 'RealAmplitudes':
            self.ansatz = RealAmplitudes(
                num_qubits=self.n_features,
                reps=ansatz_reps,
                entanglement=entanglement
            )
        elif ansatz_type == 'TwoLocal':
            self.ansatz = TwoLocal(
                num_qubits=self.n_features,
                rotation_blocks='ry',
                entanglement_blocks='cz',
                reps=ansatz_reps,
                entanglement=entanglement
            )
        else:
            raise ValueError(f"Unknown ansatz type: {ansatz_type}")
        
        # Compose circuit
        self.circuit = QuantumCircuit(self.n_features)
        self.circuit.compose(self.feature_map, inplace=True)
        self.circuit.compose(self.ansatz, inplace=True)
        
        # Transpile if using IBM backend
        self.circuit = self.quantum_backend.transpile_circuit(self.circuit)
        
        logger.info(f"Built VQC circuit: {self.n_features} qubits, "
                   f"{self.circuit.depth()} depth")
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'QuantumVQCModel':
        """Train VQC."""
        logger.info("Training Quantum VQC...")
        start_time = time.time()
        
        # Validate feature dimension
        if X.shape[1] != self.n_features:
            raise ValueError(f"Expected {self.n_features} features, got {X.shape[1]}")
        
        # Get estimator
        estimator = self.quantum_backend.get_estimator()
        
        # Create QNN
        qnn = EstimatorQNN(
            circuit=self.circuit,
            estimator=estimator,
            input_params=self.feature_map.parameters,
            weight_params=self.ansatz.parameters
        )
        
        # Get optimizer
        opt_config = self.vqc_config.get('optimizer', {})
        opt_type = opt_config.get('type', 'COBYLA')
        maxiter = opt_config.get('maxiter', 100)
        
        if opt_type == 'COBYLA':
            optimizer = COBYLA(maxiter=maxiter)
        elif opt_type == 'SPSA':
            optimizer = SPSA(maxiter=maxiter)
        elif opt_type == 'Adam':
            optimizer = Adam(maxiter=maxiter)
        else:
            raise ValueError(f"Unknown optimizer: {opt_type}")
        
        # Create classifier
        self.model = NeuralNetworkClassifier(
            neural_network=qnn,
            optimizer=optimizer,
            one_hot=False
        )
        
        # Train
        self.model.fit(X, y)
        
        self.training_time = time.time() - start_time
        self.is_fitted = True
        
        logger.info(f"Quantum VQC trained in {self.training_time:.2f}s")
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        # VQC may not have predict_proba, so we'll handle this gracefully
        try:
            return self.model.predict_proba(X)
        except AttributeError:
            # Return prediction as probability
            pred = self.model.predict(X)
            proba = np.zeros((len(pred), 2))
            proba[pred == 0, 0] = 1.0
            proba[pred == 1, 1] = 1.0
            return proba


class QuantumSVMModel(BaseModel):
    """Quantum Support Vector Machine with quantum kernel."""
    
    def __init__(self, config: Dict[str, Any], n_features: int):
        super().__init__(config)
        self.n_features = n_features
        self.backend_config = config.get('backend', {})
        self.qsvm_config = config.get('qsvm', {})
        self.quantum_backend = QuantumBackend(self.backend_config)
        self._build_feature_map()
    
    def _build_feature_map(self):
        """Build feature map for quantum kernel."""
        fm_config = self.qsvm_config.get('feature_map', {})
        fm_type = fm_config.get('type', 'ZZFeatureMap')
        fm_reps = fm_config.get('reps', 2)
        entanglement = fm_config.get('entanglement', 'linear')
        
        if fm_type == 'ZZFeatureMap':
            self.feature_map = ZZFeatureMap(
                feature_dimension=self.n_features,
                reps=fm_reps,
                entanglement=entanglement
            )
        elif fm_type == 'ZFeatureMap':
            self.feature_map = ZFeatureMap(
                feature_dimension=self.n_features,
                reps=fm_reps
            )
        else:
            raise ValueError(f"Unknown feature map type: {fm_type}")
        
        # Transpile feature map
        self.feature_map = self.quantum_backend.transpile_circuit(self.feature_map)
        
        logger.info(f"Built quantum feature map for kernel: {self.n_features} qubits")
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'QuantumSVMModel':
        """Train Quantum SVM."""
        logger.info("Training Quantum SVM...")
        start_time = time.time()
        
        # Validate feature dimension
        if X.shape[1] != self.n_features:
            raise ValueError(f"Expected {self.n_features} features, got {X.shape[1]}")
        
        # Get sampler
        sampler = self.quantum_backend.get_sampler()
        
        # Create quantum kernel
        fidelity = ComputeUncompute(sampler=sampler)
        quantum_kernel = FidelityQuantumKernel(
            feature_map=self.feature_map,
            fidelity=fidelity
        )
        
        # Create SVM with quantum kernel
        self.model = SVC(
            kernel=quantum_kernel.evaluate,
            C=self.qsvm_config.get('C', 1.0),
            class_weight=self.qsvm_config.get('class_weight', 'balanced'),
            probability=True,
            cache_size=self.qsvm_config.get('cache_size', 1000),
            max_iter=self.qsvm_config.get('max_iter', -1)
        )
        
        # Train
        self.model.fit(X, y)
        
        self.training_time = time.time() - start_time
        self.is_fitted = True
        
        logger.info(f"Quantum SVM trained in {self.training_time:.2f}s")
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        return self.model.predict_proba(X)


# ==============================================================================
# MODEL FACTORY
# ==============================================================================

def create_model(
    model_name: str,
    config: Dict[str, Any],
    n_features: Optional[int] = None
) -> BaseModel:
    """
    Factory function to create models.
    
    Args:
        model_name: Name of model (e.g., 'logistic_regression', 'quantum_vqc')
        config: Model configuration
        n_features: Number of input features (required for quantum models)
    
    Returns:
        Model instance
    """
    model_map = {
        'logistic_regression': LogisticRegressionModel,
        'random_forest': RandomForestModel,
        'xgboost': XGBoostModel,
        'quantum_vqc': QuantumVQCModel,
        'quantum_svm': QuantumSVMModel,
    }
    
    if model_name not in model_map:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(model_map.keys())}")
    
    model_class = model_map[model_name]
    
    # Quantum models require n_features
    if model_name in ['quantum_vqc', 'quantum_svm']:
        if n_features is None:
            raise ValueError(f"{model_name} requires n_features parameter")
        return model_class(config, n_features)
    else:
        return model_class(config)
