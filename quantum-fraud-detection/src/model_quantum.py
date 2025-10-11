from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Literal
import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit.library import ZZFeatureMap, TwoLocal
from qiskit.primitives import Estimator, Sampler
from qiskit_algorithms.optimizers import COBYLA
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from sklearn.svm import SVC

from .quantum_backend import BackendConfig, get_estimator, get_sampler


@dataclass
class QuantumConfig:
    num_features: int
    reps_feature_map: int = 2
    reps_ansatz: int = 2
    optimizer_maxiter: int = 100
    shots: Optional[int] = None  # None -> default, else used in Estimator options
    initial_point_seed: int = 42
    backend_config: Optional[BackendConfig] = None


def build_vqc(cfg: QuantumConfig) -> NeuralNetworkClassifier:
    """Build Variational Quantum Classifier.
    
    Args:
        cfg: Quantum model configuration
        
    Returns:
        Qiskit NeuralNetworkClassifier with EstimatorQNN
    """
    # Create feature map and ansatz
    feature_map = ZZFeatureMap(cfg.num_features, reps=cfg.reps_feature_map)
    ansatz = TwoLocal(cfg.num_features, rotation_blocks="ry", entanglement_blocks="cz", reps=cfg.reps_ansatz)

    # Compose them into a single circuit
    qc = QuantumCircuit(cfg.num_features)
    qc.compose(feature_map, inplace=True)
    qc.compose(ansatz, inplace=True)

    # Create estimator based on backend configuration
    if cfg.backend_config:
        estimator = get_estimator(cfg.backend_config)
    else:
        estimator = Estimator(options={"shots": cfg.shots} if cfg.shots else None)

    # Create QNN with the composed circuit
    qnn = EstimatorQNN(
        circuit=qc,
        estimator=estimator,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
    )

    optimizer = COBYLA(maxiter=cfg.optimizer_maxiter)

    classifier = NeuralNetworkClassifier(
        neural_network=qnn,
        optimizer=optimizer,
        one_hot=False,
        callback=None,
    )
    return classifier


def train_vqc(X_train: np.ndarray, y_train: np.ndarray, cfg: QuantumConfig) -> NeuralNetworkClassifier:
    """Train Variational Quantum Classifier.
    
    Args:
        X_train: Training features (numpy array)
        y_train: Training labels (numpy array)
        cfg: Quantum model configuration
        
    Returns:
        Trained quantum classifier
        
    Raises:
        ValueError: If feature dimension mismatch
    """
    if X_train.shape[1] != cfg.num_features:
        raise ValueError(f"X_train has {X_train.shape[1]} features, expected {cfg.num_features}")
    clf = build_vqc(cfg)
    clf.fit(X_train, y_train)
    return clf


@dataclass
class QuantumKernelConfig:
    num_features: int
    reps_feature_map: int = 2
    shots: Optional[int] = None
    C: float = 1.0  # SVM regularization parameter
    gamma: str = "scale"  # SVM kernel coefficient
    backend_config: Optional[BackendConfig] = None


def build_quantum_kernel(cfg: QuantumKernelConfig) -> SVC:
    """Build Quantum Kernel SVM classifier.
    
    Args:
        cfg: Quantum Kernel configuration
        
    Returns:
        SVC with quantum kernel
    """
    # Create feature map for quantum kernel
    feature_map = ZZFeatureMap(cfg.num_features, reps=cfg.reps_feature_map)
    
    # Create sampler based on backend configuration
    if cfg.backend_config:
        sampler = get_sampler(cfg.backend_config)
    else:
        sampler = Sampler(options={"shots": cfg.shots} if cfg.shots else None)
    
    # Create quantum kernel
    quantum_kernel = FidelityQuantumKernel(feature_map=feature_map, sampler=sampler)
    
    # Create SVC with quantum kernel
    svc = SVC(kernel=quantum_kernel.evaluate, C=cfg.C, probability=True)
    
    return svc


def train_quantum_kernel(X_train: np.ndarray, y_train: np.ndarray, cfg: QuantumKernelConfig) -> SVC:
    """Train Quantum Kernel SVM classifier.
    
    Args:
        X_train: Training features (numpy array)
        y_train: Training labels (numpy array)
        cfg: Quantum Kernel configuration
        
    Returns:
        Trained quantum kernel SVM classifier
        
    Raises:
        ValueError: If feature dimension mismatch
    """
    if X_train.shape[1] != cfg.num_features:
        raise ValueError(f"X_train has {X_train.shape[1]} features, expected {cfg.num_features}")
    
    clf = build_quantum_kernel(cfg)
    clf.fit(X_train, y_train)
    return clf
