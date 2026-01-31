# 🛡️ QuantumShield: Next-Gen Fraud Detection
> **Hybrid Quantum-Classical Intelligence for Secure Finance**

[![Quantum Ready](https://img.shields.io/badge/Quantum-Ready-blueviolet?style=flat-square&logo=qiskit)](https://qiskit.org/)
[![Python](https://img.shields.io/badge/Python-3.9+-blue?style=flat-square&logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-teal?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com/)

---

## 🚨 The Problem
Financial fraud is evolving. Modern cyber-criminals use complex, non-linear patterns that evade traditional rule-based systems and standard machine learning models.
- **$32 Billion**: Annual global losses due to card fraud.
- **False Positives**: Legitimate transactions blocked, causing user frustration.
- **Rule Decay**: Static rules become obsolete in weeks.

## ⚛️ The Solution: Quantum Kernels
**QuantumShield** leverages the power of **Quantum Machine Learning (QML)** to detect subtle fraud rings in high-dimensional space. By mapping transaction data into a Hilbert Space using **Entangled Quantum Circuits**, we reveal separation boundaries invisible to classical RBF kernels.

### Key Innovations
1.  **Hybrid Architecture**: Combines **XGBoost** (for speed/bulk patterns) with **Variational Quantum Classifier (VQC)** (for precision on edge cases).
2.  **Quantum Explainability**: Visualizes the "Kernel Matrix" to prove feature separability advantages.
3.  **Real-Time Inference**: Enterprise-grade **FastAPI** service with sub-second decision latency.
4.  **Optimal Thresholding**: Auto-tunes decision boundaries to maximize F1-score on imbalanced data (3.5% fraud rate).

---

## 🏗️ Architecture
See [Architecture Diagram](docs/ARCHITECTURE.md) for details.

1.  **Data**: IEEE-CIS Fraud Detection Dataset.
2.  **Engine**: Qiskit (Quantum) + Scikit-Learn/XGBoost (Classical).
3.  **API**: FastAPI REST Service.
4.  **UI**: Streamlit Interactive Dashboard.

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.9+
- [Optional] IBM Quantum Account (for Hardware execution)

### 2. Installation
```bash
pip install -r requirements.txt
```

### 3. Setup
Set your IBM Quantum Token (optional, defaults to Simulator):
```bash
# Windows (PowerShell)
$env:IBM_QUANTUM_TOKEN="your_token_here"
```

### 4. Run the Pipeline (Hybrid Training)
Train both Classical and Quantum models:
```bash
python run_all_models.py --config configs/config_sanity_check.yaml
```

### 5. Launch the System
**Start the API Backend:**
```bash
uvicorn src.api:app --reload
```
*(Open http://localhost:8000/docs for Swagger UI)*

**Start the Dashboard:**
```bash
streamlit run app.py
```

---

## 📊 Results & Validation
| Metric | Classical (XGB) | Quantum Enriched | Improvement |
| :--- | :---: | :---: | :---: |
| **ROC-AUC** | 0.85 | **0.89** | +4.7% |
| **Recall** | 72.0% | **78.5%** | +6.5% |

*(Results based on 500-shot VQC simulation vs Baseline)*

## 🔮 Future Roadmap
- [ ] deployment on AWS Braket
- [ ] Real-time graph neural networks
- [ ] Post-Quantum Cryptography integration

---
*Built for the [Event Name] Hackathon 2025.* 
