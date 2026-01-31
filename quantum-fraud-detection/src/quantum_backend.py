"""Quantum backend management for simulator and IBM Quantum hardware."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Literal
import logging

from qiskit.primitives import Estimator, Sampler
from qiskit_aer.primitives import Estimator as AerEstimator, Sampler as AerSampler

try:
    from qiskit_ibm_runtime import QiskitRuntimeService, Estimator as IBMEstimator, SamplerV2 as IBMSampler
    IBM_AVAILABLE = True
except ImportError:
    IBM_AVAILABLE = False
    logging.warning("qiskit-ibm-runtime not installed. IBM Quantum hardware support disabled.")

try:
    from qiskit.primitives import StatevectorSampler
    V2_AVAILABLE = True
except ImportError:
    V2_AVAILABLE = False
    logging.warning("qiskit 1.0+ V2 primitives not found. Some functionality might fail.")


@dataclass
class BackendConfig:
    """Configuration for quantum backend selection."""
    backend_type: Literal["simulator", "aer", "ibm_quantum"] = "simulator"
    ibm_token: Optional[str] = None
    ibm_backend_name: Optional[str] = None  # e.g., "ibmq_qasm_simulator", "ibm_brisbane"
    shots: Optional[int] = None
    optimization_level: int = 1
    resilience_level: int = 1  # 0=No mitigation, 1=TREM


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
        run_options = {"shots": cfg.shots} if cfg.shots else None
        return AerEstimator(run_options=run_options)
    
    elif cfg.backend_type == "ibm_quantum":
        if not IBM_AVAILABLE:
            raise ValueError("qiskit-ibm-runtime not installed. Install with: pip install qiskit-ibm-runtime")
        
        # Auto-load from environment if not provided
        if not cfg.ibm_token:
            import os
            cfg.ibm_token = os.environ.get("IBM_QUANTUM_TOKEN")

        if not cfg.ibm_token:
            raise ValueError(
                "CRITICAL: IBM Quantum Token is MISSING for 'ibm_quantum' backend.\n"
                "Please set it in 'configs/config.yaml' (ibm_token) or via environment variable 'IBM_QUANTUM_TOKEN'.\n"
                "Get your token from: https://quantum.ibm.com/"
            )
        
        # Initialize IBM Quantum service
        # 'ibm_quantum' channel might be deprecated or renamed in some versions.
        # Letting it infer from account or using default.
        try:
             service = QiskitRuntimeService(channel="ibm_quantum", token=cfg.ibm_token)
        except ValueError:
             # Fallback for newer/different versions complaining about channel name
             logging.warning("Retrying with channel='ibm_quantum' failed (or similar). Trying without channel arg.")
             service = QiskitRuntimeService(token=cfg.ibm_token)
        
        # Get backend
        backend_name = cfg.ibm_backend_name or "ibmq_qasm_simulator"
        backend = service.backend(backend_name)
        
        logging.info(f"Using IBM Quantum backend: {backend_name}")
        
        # Create IBM Estimator
        options = {
            "optimization_level": cfg.optimization_level,
            "resilience_level": cfg.resilience_level
        }
        if cfg.shots:
            options["shots"] = cfg.shots
        
        # Create IBM Estimator
        # EstimatorV2 uses 'mode' instead of 'backend' and 'default_shots' instead of 'shots'
        
        try:
             # Try V2 style options construction
             # Use dict with V2 keys
             v2_options = {}
             if cfg.shots:
                 v2_options["default_shots"] = cfg.shots
             
             # resilience/optimization levels might not map directly in V2 or need specific structure
             # For safety/compatibility, we only set shots which is critical
             
             return IBMEstimator(mode=backend, options=v2_options)
        except TypeError:
             # Fallback to V1 style if older version
             return IBMEstimator(backend=backend, options=options)
    
    else:
        raise ValueError(f"Unknown backend type: {cfg.backend_type}")


def get_sampler(cfg: BackendConfig):
    """Get appropriate Sampler based on backend configuration.
    
    Args:
        cfg: Backend configuration
        
    Returns:
        Sampler instance (V2 if possible)
        
    Raises:
        ValueError: If IBM backend requested but not available
    """
    if cfg.backend_type == "simulator":
        if V2_AVAILABLE:
            # Use V2 Local Simulator
            # StatevectorSampler(default_shots=...)
            default_shots = cfg.shots if cfg.shots else None
            try:
                sampler = StatevectorSampler(default_shots=default_shots)
            except TypeError:
                # In case older version doesn't accept it, try without
                logging.warning("StatevectorSampler does not accept default_shots in init. Using defaults.")
                sampler = StatevectorSampler()
            return sampler
        else:
            # Fallback to V1 (might fail with ComputeUncompute)
            options = {"shots": cfg.shots} if cfg.shots else None
            return Sampler(options=options)
    
    elif cfg.backend_type == "aer":
        # Aer not fully V2 compliant in all versions, trying basic V1 for now unless requested
        # Or check if AerSamplerV2 exists
        try:
             from qiskit_aer.primitives import SamplerV2
             sampler = SamplerV2()
             if cfg.shots:
                 sampler.default_shots = cfg.shots
             return sampler
        except ImportError:
             # Use Aer simulator V1
             run_options = {"shots": cfg.shots} if cfg.shots else None
             return AerSampler(run_options=run_options)
    
    elif cfg.backend_type == "ibm_quantum":
        if not IBM_AVAILABLE:
            raise ValueError("qiskit-ibm-runtime not installed. Install with: pip install qiskit-ibm-runtime")
        
        # Auto-load from environment if not provided
        if not cfg.ibm_token:
            import os
            cfg.ibm_token = os.environ.get("IBM_QUANTUM_TOKEN")

        if not cfg.ibm_token:
            raise ValueError(
                "CRITICAL: IBM Quantum Token is MISSING for 'ibm_quantum' backend.\n"
                "Please set it in 'configs/config.yaml' (ibm_token) or via environment variable 'IBM_QUANTUM_TOKEN'.\n"
                "Get your token from: https://quantum.ibm.com/"
            )
        
        # Initialize IBM Quantum service
        try:
            service = QiskitRuntimeService(channel="ibm_quantum", token=cfg.ibm_token)
        except ValueError:
            service = QiskitRuntimeService(token=cfg.ibm_token)
        
        # Get backend
        backend_name = cfg.ibm_backend_name or "ibmq_qasm_simulator"
        backend = service.backend(backend_name)
        
        logging.info(f"Using IBM Quantum backend (SamplerV2): {backend_name}")
        
        # Create IBM SamplerV2
        # mode argument is indeed the backend in recent versions
        sampler = IBMSampler(mode=backend)
        if cfg.shots:
            sampler.options.default_shots = cfg.shots
            
        return sampler
    
    else:
        raise ValueError(f"Unknown backend type: {cfg.backend_type}")


def transpile_circuit(circuit, cfg: BackendConfig, scrub_parameters: bool = False):
    """Transpile circuit for the target backend, ensuring ISA compliance.
    
    Args:
        circuit: Qiskit QuantumCircuit
        cfg: Backend configuration
        scrub_parameters: If True, replaces parameters with fresh ParameterVector (fixes QPY name conflicts)
        
    Returns:
        Transpiled QuantumCircuit
    """
    if cfg.backend_type != "ibm_quantum":
        return circuit

    if not IBM_AVAILABLE:
        logging.warning("Cannot transpile for IBM Backend: qiskit-ibm-runtime missing.")
        return circuit

    try:
        # Initialize Service
        try:
             # Try simple init first (env var or default)
             if not cfg.ibm_token:
                 import os
                 cfg.ibm_token = os.environ.get("IBM_QUANTUM_TOKEN")

             if cfg.ibm_token:
                 try:
                    service = QiskitRuntimeService(channel="ibm_quantum", token=cfg.ibm_token)
                 except ValueError:
                    service = QiskitRuntimeService(token=cfg.ibm_token)
             else:
                 service = QiskitRuntimeService()
        except Exception:
             # Fallback
             service = QiskitRuntimeService()

        backend_name = cfg.ibm_backend_name or "ibmq_qasm_simulator"
        backend = service.backend(backend_name)
        
        logging.info(f"Transpiling circuit for backend: {backend_name}")
        
        # Import transpile here to avoid circular dependencies if any
        from qiskit import transpile
        
        # Optimization Level 3 required for some gates (like sxdg) on new backends
        try:
            transpiled_qc = transpile(circuit, target=backend.target, optimization_level=3)
        except Exception:
             transpiled_qc = transpile(circuit, backend=backend, optimization_level=3)
             
        if scrub_parameters:
            from qiskit.circuit import ParameterVector
            logging.info("Scrubbing parameters to prevent StateFidelity name conflicts.")
            # Create fresh parameters matching the count
            safe_params = ParameterVector("safe_x", transpiled_qc.num_parameters)
            old_params = transpiled_qc.parameters
            
            if len(safe_params) == len(old_params):
                param_map = {old: new for old, new in zip(old_params, safe_params)}
                transpiled_qc.assign_parameters(param_map, inplace=True)
            else:
                logging.warning("Parameter count mismatch during scrubbing. Skipping.")
                
        return transpiled_qc

    except Exception as e:
        logging.warning(f"Transpilation failed: {e}. Returning original circuit.")
        return circuit
