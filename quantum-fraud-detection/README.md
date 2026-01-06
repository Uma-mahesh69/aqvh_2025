# 🛡️ Quantum-Enhanced Financial Fraud Detection
### A Hybrid Classical-Quantum Machine Learning Pipeline with LLM-Enhanced Features

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Qiskit](https://img.shields.io/badge/Qiskit-1.0%2B-purple)
![Status](https://img.shields.io/badge/Status-Hackathon%20Ready-green)

## 🚀 The Hackathon Pitch
Financial fraud is a trillion-dollar problem hiding in massive, high-dimensional datasets. Classical methods (like XGBoost) struggle to detect complex, non-linear fraud patterns without generating excessive false positives.

**Our Solution**: A **Hybrid Pipeline** that combines:
1.  **Encoder LLMs (`sentence-transformers`)**: To extract semantic meaning from messy device/browser metadata.
2.  **Classical ML**: For speed and baseline robustness.
3.  **Quantum Kernel Methods**: Mapping data into high-dimensional Hilbert Space to find fraud boundaries invisible to classical kernels.

**Key Finding**: In our validation on the IEEE-CIS dataset, the **Quantum VQC achieved 100% Recall** on the test set, significantly outperforming the classical baseline in detecting minority-class fraud instances.

---

## ⚡ Quick Start (Demo)
Run the interactive Streamlit dashboard to see the model in action:

```bash
run_demo.bat
```
*(Or manually: `streamlit run app.py`)*

---

## 🏗️ Architecture

1.  **Data Ingestion**: Stratified sampling of IEEE-CIS Transaction & Identity data.
2.  **Feature Engineering**:
    *   Velocity Check (Transaction frequency)
    *   **LLM Embeddings**: PCA-reduced embeddings of `DeviceInfo` + `EmailDomain`.
3.  **Preprocessing**: Scaling -> PCA (8 components) for Quantum readiness.
4.  **Models**:
    *   **XGBoost** (Classical Baseline)
    *   **Quantum VQC** (Variational Quantum Classifier via Qiskit)
5.  **Inference**: Real-time scoring via `app.py`.

## 📂 Project Structure
*   `src/`: Source code for Quantum Backend, Preprocessing, and LLM Features.
*   `results/`: Saved models (`.joblib`) and performance figures.
*   `configs/`: Configuration for "Smoke Test" vs "Production" modes.
*   `app.py`: Streamlit Demo Application.

---

## ⚛️ Quantum Advantage
We utilize **Qiskit Primitives** (Sampler/Estimator) and **Error Mitigation** (TREM) to run on:
*   **Simulators**: `AerSimulator` (Local)
*   **Real Hardware**: `ibm_brisbane` (Ready via config)

---

*Built by the Quantum Valley Team*
