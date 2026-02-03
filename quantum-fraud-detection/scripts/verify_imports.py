
import sys
import os
import logging
sys.path.append(os.getcwd())

logging.basicConfig(level=logging.INFO)

def check_imports():
    print("Checking imports...")
    try:
        from src.pipeline_manager import FraudDetectionPipeline
        from src.preprocessing import preprocess_pipeline
        from src.inference import FraudInference
        from src.model_quantum import train_vqc
        from src.explainability import compute_and_plot_kernels
        from src.quantum_backend import BackendConfig
        import app
        print("✅ All modules imported successfully.")
    except Exception as e:
        print(f"❌ Import Failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    check_imports()
