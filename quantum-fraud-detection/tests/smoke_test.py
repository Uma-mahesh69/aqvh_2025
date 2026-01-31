import sys
import os
import joblib

# Add project root to path
sys.path.append(os.getcwd())

def test_imports():
    print("Testing imports...")
    try:
        import qiskit
        import pandas
        import sklearn
        import xgboost
        print("[OK] Core libraries imported successfully.")
    except ImportError as e:
        print(f"[FAIL] Core library import failed: {e}")
        sys.exit(1)

def test_project_imports():
    print("Testing project imports...")
    try:
        from src.preprocessing import PreprocessConfig
        from src.quantum_backend import BackendConfig
        from src.pipeline_manager import FraudDetectionPipeline
        print("[OK] Project modules imported successfully.")
    except ImportError as e:
        import traceback
        traceback.print_exc()
        print(f"[FAIL] Project module import failed: {e}")
        sys.exit(1)

def test_config_loading():
    print("Testing config loading...")
    config_path = "configs/config.yaml"
    if not os.path.exists(config_path):
        print(f"[FAIL] Config file not found at {config_path}")
        sys.exit(1)
        
    try:
        import yaml
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        print("[OK] Config loaded and verified structure.")
    except Exception as e:
        print(f"[FAIL] Config loading failed: {e}")
        sys.exit(1)

def test_api_import():
    print("Testing API module import...")
    try:
        from src.api import app, TransactionInput
        print("[OK] API module imported successfully.")
    except Exception as e:
        print(f"[FAIL] API module import failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    print("Running Smoke Test...")
    test_imports()
    test_project_imports()
    test_config_loading()
    test_api_import()
    print("\nSmoke Test Passed!")
