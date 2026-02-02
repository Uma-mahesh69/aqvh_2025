from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Literal
import numpy as np
import logging

from qiskit import QuantumCircuit
from qiskit.circuit.library import ZZFeatureMap, TwoLocal, ZFeatureMap
from qiskit_algorithms.optimizers import COBYLA, SPSA
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_algorithms.state_fidelities import ComputeUncompute
from sklearn.svm import SVC

from .quantum_backend import BackendConfig, get_estimator, get_sampler, transpile_circuit


@dataclass
class QuantumConfig:
    num_features: int
    reps_feature_map: int = 2
    reps_ansatz: int = 2
    optimizer_maxiter: int = 100
    shots: Optional[int] = None
    optimizer: str = "cobyla" # cobyla, spsa
    backend_config: Optional[BackendConfig] = None


def build_vqc(cfg: QuantumConfig) -> NeuralNetworkClassifier:
    """Build Variational Quantum Classifier."""
    
    # 1. Feature Map & Ansatz
    # Use ZFeatureMap for hardware efficiency if on IBM, else ZZ
    if cfg.backend_config and cfg.backend_config.backend_type == "ibm_quantum":
        feature_map = ZFeatureMap(cfg.num_features, reps=cfg.reps_feature_map)
    else:
        feature_map = ZZFeatureMap(cfg.num_features, reps=cfg.reps_feature_map)
        
    ansatz = TwoLocal(
        cfg.num_features, 
        rotation_blocks="ry", 
        entanglement_blocks="cz", 
        entanglement="linear",
        reps=cfg.reps_ansatz
    )

    # 2. Compose Circuit
    qc = QuantumCircuit(cfg.num_features)
    qc.compose(feature_map, inplace=True)
    qc.compose(ansatz, inplace=True)

    # 3. Transpile if needed (crucial for IBM ISA)
    if cfg.backend_config:
        qc = transpile_circuit(qc, cfg.backend_config, scrub_parameters=False)

    # 4. Estimator
    estimator = get_estimator(cfg.backend_config or BackendConfig())

    # 5. QNN
    qnn = EstimatorQNN(
        circuit=qc,
        estimator=estimator,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
    )

    # 6. Optimizer
    if cfg.optimizer.lower() == "spsa":
        optimizer = SPSA(maxiter=cfg.optimizer_maxiter)
    else:
        optimizer = COBYLA(maxiter=cfg.optimizer_maxiter)

    # 7. Classifier
    classifier = NeuralNetworkClassifier(
        neural_network=qnn,
        optimizer=optimizer,
        one_hot=False
    )
    return classifier


def train_vqc(X_train: np.ndarray, y_train: np.ndarray, cfg: QuantumConfig) -> NeuralNetworkClassifier:
    """Train VQC."""
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
    C: float = 1.0
    gamma: str = "scale"
    backend_config: Optional[BackendConfig] = None


def build_quantum_kernel(cfg: QuantumKernelConfig) -> SVC:
    """Build Quantum Kernel SVM."""
    
    # 1. Feature Map
    if cfg.backend_config and cfg.backend_config.backend_type == "ibm_quantum":
        feature_map = ZFeatureMap(cfg.num_features, reps=cfg.reps_feature_map)
    else:
        feature_map = ZZFeatureMap(cfg.num_features, reps=cfg.reps_feature_map)
    
    # 2. Transpile & Scrub (Scrubbing often needed for Kernels if params overlap)
    if cfg.backend_config:
        feature_map = transpile_circuit(
            feature_map, 
            cfg.backend_config, 
            scrub_parameters=True 
        )

    # 3. Sampler & Fidelity
    sampler = get_sampler(cfg.backend_config or BackendConfig())
    
    # Note: ComputeUncompute might need explicit V1/V2 check in future.
    # Currently assuming it handles the sampler provided.
    try:
        fidelity = ComputeUncompute(sampler=sampler)
    except Exception as e:
        logging.warning("ComputeUncompute failed with provided sampler. Trying fallback.")
        raise e

    # 4. Kernel
    quantum_kernel = FidelityQuantumKernel(feature_map=feature_map, fidelity=fidelity)
    
    # 5. SVC
    svc = SVC(kernel=quantum_kernel.evaluate, C=cfg.C, probability=True)
    return svc


def train_quantum_kernel(X_train: np.ndarray, y_train: np.ndarray, cfg: QuantumKernelConfig) -> SVC:
    """Train Quantum Kernel SVM."""
    if X_train.shape[1] != cfg.num_features:
        raise ValueError(f"X_train has {X_train.shape[1]} features, expected {cfg.num_features}")
    
    clf = build_quantum_kernel(cfg)
    clf.fit(X_train, y_train)
    return clf
