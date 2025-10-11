"""Quantum backend management for simulator and IBM Quantum hardware."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Literal
import logging

from qiskit.primitives import Estimator, Sampler
from qiskit_aer.primitives import Estimator as AerEstimator, Sampler as AerSampler

try:
    from qiskit_ibm_runtime import QiskitRuntimeService, Estimator as IBMEstimator, Sampler as IBMSampler
    IBM_AVAILABLE = True
except ImportError:
    IBM_AVAILABLE = False
    logging.warning("qiskit-ibm-runtime not installed. IBM Quantum hardware support disabled.")


@dataclass
class BackendConfig:
    """Configuration for quantum backend selection."""
    backend_type: Literal["simulator", "aer", "ibm_quantum"] = "simulator"
    ibm_token: Optional[str] = None
    ibm_backend_name: Optional[str] = None  # e.g., "ibmq_qasm_simulator", "ibm_brisbane"
    shots: Optional[int] = None
    optimization_level: int = 1


def get_estimator(cfg: BackendConfig):
    """Get appropriate Estimator based on backend configuration.
    
    Args:
        cfg: Backend configuration
        
    Returns:
        Estimator instance (local simulator, Aer, or IBM)
        
    Raises:
        ValueError: If IBM backend requested but not available
    """
    if cfg.backend_type == "simulator":
        # Use basic local simulator
        options = {"shots": cfg.shots} if cfg.shots else None
        return Estimator(options=options)
    
    elif cfg.backend_type == "aer":
        # Use Aer simulator (more realistic noise models)
        options = {"shots": cfg.shots} if cfg.shots else None
        return AerEstimator(options=options)
    
    elif cfg.backend_type == "ibm_quantum":
        if not IBM_AVAILABLE:
            raise ValueError("qiskit-ibm-runtime not installed. Install with: pip install qiskit-ibm-runtime")
        
        if not cfg.ibm_token:
            raise ValueError("IBM Quantum token required for IBM backend")
        
        # Initialize IBM Quantum service
        service = QiskitRuntimeService(channel="ibm_quantum", token=cfg.ibm_token)
        
        # Get backend
        backend_name = cfg.ibm_backend_name or "ibmq_qasm_simulator"
        backend = service.backend(backend_name)
        
        logging.info(f"Using IBM Quantum backend: {backend_name}")
        
        # Create IBM Estimator
        options = {"optimization_level": cfg.optimization_level}
        if cfg.shots:
            options["shots"] = cfg.shots
        
        return IBMEstimator(backend=backend, options=options)
    
    else:
        raise ValueError(f"Unknown backend type: {cfg.backend_type}")


def get_sampler(cfg: BackendConfig):
    """Get appropriate Sampler based on backend configuration.
    
    Args:
        cfg: Backend configuration
        
    Returns:
        Sampler instance (local simulator, Aer, or IBM)
        
    Raises:
        ValueError: If IBM backend requested but not available
    """
    if cfg.backend_type == "simulator":
        # Use basic local simulator
        options = {"shots": cfg.shots} if cfg.shots else None
        return Sampler(options=options)
    
    elif cfg.backend_type == "aer":
        # Use Aer simulator
        options = {"shots": cfg.shots} if cfg.shots else None
        return AerSampler(options=options)
    
    elif cfg.backend_type == "ibm_quantum":
        if not IBM_AVAILABLE:
            raise ValueError("qiskit-ibm-runtime not installed. Install with: pip install qiskit-ibm-runtime")
        
        if not cfg.ibm_token:
            raise ValueError("IBM Quantum token required for IBM backend")
        
        # Initialize IBM Quantum service
        service = QiskitRuntimeService(channel="ibm_quantum", token=cfg.ibm_token)
        
        # Get backend
        backend_name = cfg.ibm_backend_name or "ibmq_qasm_simulator"
        backend = service.backend(backend_name)
        
        logging.info(f"Using IBM Quantum backend: {backend_name}")
        
        # Create IBM Sampler
        options = {"optimization_level": cfg.optimization_level}
        if cfg.shots:
            options["shots"] = cfg.shots
        
        return IBMSampler(backend=backend, options=options)
    
    else:
        raise ValueError(f"Unknown backend type: {cfg.backend_type}")
