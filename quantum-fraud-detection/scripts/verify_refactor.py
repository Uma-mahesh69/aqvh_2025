
import sys
import os
sys.path.append(os.getcwd())

import numpy as np
from src.quantum_backend import BackendConfig, get_estimator
from src.model_quantum import QuantumConfig, build_vqc

def test_backend_setup():
    print("Testing Backend Setup...")
    
    # 1. Config
    backend_cfg = BackendConfig(backend_type="simulator", shots=100)
    q_cfg = QuantumConfig(num_features=4, backend_config=backend_cfg, optimizer="cobyla")
    
    # 2. Build VQC
    print("Building VQC...")
    vqc = build_vqc(q_cfg)
    print("VQC Built Successfully.")
    
    # 3. Dummy Train
    print("Dummy Training...")
    X = np.random.rand(5, 4)
    y = np.array([0, 1, 0, 1, 0])
    vqc.fit(X, y)
    print("Verification Complete: Training runs without error.")

if __name__ == "__main__":
    test_backend_setup()
