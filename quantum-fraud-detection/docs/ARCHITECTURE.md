# QuantumShield Architecture

## 🏗️ System Overview (One-Slide)

```mermaid
graph TD
    subgraph Data Layer
        A[IEEE-CIS Dataset] -->|Load & Clean| B(Preprocessing Engine)
        B -->|SMOTE & Scaling| C{Feature Engineering}
        C -->|PCA / Reduction| D[Optimized Features]
    end

    subgraph Hybrid Model Layer
        D --> E[Classical XGBoost]
        D --> F[Quantum Feature Map]
        F -->|Hilbert Space Projection| G[Variational Quantum Classifier]
        E -->|Logits| H[Ensemble Logic]
        G -->|Probabilities| H
    end

    subgraph Deployment Layer
        H -->|Risk Score| I[Inference Engine (API)]
        I -->|JSON Response| J[Streamlit Dashboard]
        I -->|REST Endpoint| K[External Banking Systems]
    end
    
    style F fill:#f9f,stroke:#333,stroke-width:2px
    style G fill:#f9f,stroke:#333,stroke-width:2px
    style I fill:#bbf,stroke:#333,stroke-width:2px
```

## 🏦 Business Impact Mapping

| Feature | Technical Implementation | Business Value |
| :--- | :--- | :--- |
| **Hybrid Detection** | XGBoost + VQC Ensemble | **Reduced False Positives**: Quantum model captures non-linear edge cases missed by classical rules. |
| **Quantum Kernels** | `ZZFeatureMap` (Entanglement) | **Uncover SOTA Patterns**: Detects complex fraud rings hidden in high-dimensional data. |
| **Real-Time API** | FastAPI + Inference Class | **Instant Decisioning**: <100ms response time for transaction blocking. |
| **Explainability** | Kernel Matrix Visualizer | **Regulatory Compliance**: Visual proof of model decision boundaries for auditors. |

## 🚀 Deployment Checklist

- [x] **Environment**: `pip install -r requirements.txt`
- [x] **Config**: Set `IBM_QUANTUM_TOKEN`
- [x] **API**: `uvicorn src.api:app --reload`
- [x] **UI**: `streamlit run app.py`
