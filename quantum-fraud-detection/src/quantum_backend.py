"""Quantum backend management for Qiskit 1.x and IBM Quantum Hardware."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Literal, Union, Any
import logging
import os

# Qiskit 1.x Imports
from qiskit import QuantumCircuit, transpile
from qiskit.primitives import StatevectorSampler, Estimator, Sampler
from qiskit.circuit import ParameterVector

# Optional: Aer
try:
    from qiskit_aer import AerSimulator
    from qiskit_aer.primitives import Estimator as AerEstimator, Sampler as AerSampler
    AER_AVAILABLE = True
except ImportError:
    AER_AVAILABLE = False

# Optional: IBM Quantum
try:
    from qiskit_ibm_runtime import QiskitRuntimeService, EstimatorV2 as IBMEstimator, SamplerV2 as IBMSampler
    from qiskit_ibm_runtime.fake_provider import FakeManilaV2
    IBM_AVAILABLE = True
except ImportError:
    IBM_AVAILABLE = False


@dataclass
class BackendConfig:
    """Configuration for quantum backend selection."""
    backend_type: Literal["simulator", "aer", "ibm_quantum"] = "simulator"
    ibm_token: Optional[str] = None
    ibm_backend_name: Optional[str] = None
    shots: Optional[int] = None
    optimization_level: int = 3
    resilience_level: int = 1


def get_service(token: Optional[str] = None) -> Any:
    """Helper to get QiskitRuntimeService."""
    if not IBM_AVAILABLE:
        raise ImportError("qiskit-ibm-runtime is not installed.")
    
    # Try grabbing from env if not provided
    if not token:
        token = os.environ.get("IBM_QUANTUM_TOKEN")
    
    try:
        if token:
            return QiskitRuntimeService(channel="ibm_quantum", token=token)
        return QiskitRuntimeService()
    except Exception as e:
        # If simple init fails, try one more time without channel (older accounts) or raise
        logging.warning(f"Standard Service init failed: {e}. Trying fallback.")
        if token:
            return QiskitRuntimeService(token=token)
        raise ValueError("Could not initialize QiskitRuntimeService. Check your token.")


def get_backend_handle(cfg: BackendConfig):
    """Refactored logic to get the actual backend object."""
    if cfg.backend_type == "simulator":
        return None # No backend object needed for local primitives usually
        
    elif cfg.backend_type == "aer":
        if not AER_AVAILABLE:
            raise ValueError("qiskit-aer is not installed.")
        return AerSimulator()
        
    elif cfg.backend_type == "ibm_quantum":
        service = get_service(cfg.ibm_token)
        backend_name = cfg.ibm_backend_name or "ibmq_qasm_simulator"
        try:
            return service.backend(backend_name)
        except Exception:
            logging.info(f"Backend {backend_name} not found, listing available...")
            available = [b.name for b in service.backends()]
            raise ValueError(f"Backend {backend_name} not found. Available: {available}")
    
    return None


def get_estimator(cfg: BackendConfig):
    """Get Qiskit Estimator (V2 preferred for IBM, V1/Standard for local)."""
    options = {}
    if cfg.shots:
        options["default_shots"] = cfg.shots # V2 naming
    
    if cfg.backend_type == "simulator":
        # Local standard Estimator (Exact)
        # Note: Standard Estimator in 1.0 doesn't take shots for exact expectation, 
        # but if we want shot noise, we might need a different approach or Aer.
        # For 'simulator' we assume exact statevector usually.
        return Estimator() 

    elif cfg.backend_type == "aer":
        if not AER_AVAILABLE:
             logging.warning("Aer not found, falling back to standard estimator.")
             return Estimator()
        
        # Aer Estimator V1 style usually
        run_options = {"shots": cfg.shots} if cfg.shots else None
        return AerEstimator(run_options=run_options)

    elif cfg.backend_type == "ibm_quantum":
        if not IBM_AVAILABLE:
            raise ValueError("IBM Runtime not installed.")
        
        backend = get_backend_handle(cfg)
        # EstimatorV2
        est = IBMEstimator(mode=backend)
        if cfg.shots:
            est.options.default_shots = cfg.shots
        
        est.options.resilience_level = cfg.resilience_level
        return est

    else:
        raise ValueError(f"Unknown backend: {cfg.backend_type}")


def get_sampler(cfg: BackendConfig):
    """Get Qiskit Sampler."""
    if cfg.backend_type == "simulator":
        # Qiskit 1.0 StatevectorSampler
        try:
            return StatevectorSampler(default_shots=cfg.shots)
        except TypeError:
             # Fallback if specific version mismatch
            s = StatevectorSampler()
            if cfg.shots: 
                # Some versions don't have default_shots in init, but in run? 
                # Actually StatevectorSampler (1.0) has default_shots in init.
                pass 
            return s

    elif cfg.backend_type == "aer":
         if not AER_AVAILABLE:
             return StatevectorSampler()
         
         run_options = {"shots": cfg.shots} if cfg.shots else None
         return AerSampler(run_options=run_options)

    elif cfg.backend_type == "ibm_quantum":
        backend = get_backend_handle(cfg)
        sampler = IBMSampler(mode=backend)
        if cfg.shots:
            sampler.options.default_shots = cfg.shots
        return sampler
        
    return StatevectorSampler()


def transpile_circuit(circuit: QuantumCircuit, cfg: BackendConfig, scrub_parameters: bool = False) -> QuantumCircuit:
    """Transpile and optionally scrub parameters."""
    
    # 1. Transpile
    if cfg.backend_type == "ibm_quantum":
        try:
             backend = get_backend_handle(cfg)
             logging.info(f"Transpiling for {backend.name}...")
             circuit = transpile(circuit, backend=backend, optimization_level=cfg.optimization_level)
        except Exception as e:
             logging.warning(f"Transpilation failed: {e}")
    
    # 2. Scrub Parameters (Fix for QPY/FeatureMap issues)
    if scrub_parameters:
        try:
            logging.info("Scrubbing parameters...")
            new_params = ParameterVector("p", circuit.num_parameters)
            current_params = circuit.parameters
            if len(new_params) == len(current_params):
                mapping = {old: new for old, new in zip(current_params, new_params)}
                circuit.assign_parameters(mapping, inplace=True)
        except Exception as e:
            logging.warning(f"Scrubbing failed: {e}")
            
    return circuit
