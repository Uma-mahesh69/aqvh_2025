# 🛡️ QuantumShield: Next-Gen Fraud Detection
> **Hybrid Quantum-Classical Intelligence for Secure Finance**

[![Quantum Ready](https://img.shields.io/badge/Quantum-Ready-blueviolet?style=flat-square&logo=qiskit)](https://qiskit.org/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-teal?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com/)

[Architecture Diagram](docs/ARCHITECTURE.md)

## 🚨 The Mission
Detect financial fraud with higher precision using **Quantum Machine Learning (QML)**. This project leverages Variational Quantum Classifiers (VQC) and Quantum Kernels to find decision boundaries invisible to classical algorithms.

## ✨ Key Features
-   **Robust Qiskit 1.0+ Support**: modernized backend handling for IBM Runtime.
-   **Hybrid Pipeline**: Co-training of XGBoost, Isolation Forest, and Quantum models.
-   **Advanced Preprocessing**: NVIDIA-inspired feature engineering (UIDs, Aggregations).
-   **Production Ready**: Integrated FastAPI service and Streamlit dashboard.

## 🛠️ Installation

```bash
# 1. Clone & Install
pip install -r requirements.txt
```

## 🚀 Usage

### 1. Train the Models
Run the full pipeline (Data -> Preprocess -> Train -> Evaluate):
```bash
python run_all_models.py
```
*Configuration*: Edit `configs/config.yaml` to enable/disable specific models (e.g., set `quantum_vqc: true` for full quantum training).

### 2. Start Inference API
Launch the REST API for real-time fraud scoring:
```bash
uvicorn src.api:app --reload
```
API Docs: [http://localhost:8000/docs](http://localhost:8000/docs)

### 3. Launch the Dashboard (Demo Mode)
Start the interactive "Mission Control" UI for the Hackathon presentation:
```bash
streamlit run app.py
```

### 4. Verification Script
Run a quick sanity check of the quantum backend:
```bash
python scripts/verify_refactor.py
```

## 🧠 Quantum Backend
The system supports:
-   **Local Simulator** (Default): Fast, noiseless simulation.
-   **IBM Quantum Hardware**: Set `IBM_QUANTUM_TOKEN` env var and configure `backend_type: "ibm_quantum"` in `config.yaml`.
-   **Aer Simulator**: Realistic noise modeling (if `qiskit-aer` is installed).

## 📊 Results will be saved to:
-   `results/metrics_table.csv`
-   `results/figures/` (ROC Curves, Confusion Matrices)
-   `results/models/` (Saved .joblib models)
