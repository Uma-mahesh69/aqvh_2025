# Quantum Fraud Detection Architecture

```mermaid
graph TD
    subgraph Data Layer
        D1[IEEE-CIS Dataset] --> L1[Smart Data Loader]
        L1 --> S1[Stratified Sampling]
    end

    subgraph Preprocessing Layer
        S1 --> P1[Feature Engineering]
        P1 --> P2[UID Generation]
        P2 --> P3[Frequency Encoding]
        P3 --> P4[SMOTE Balancing]
        P4 --> P5{Feature Selection}
        P5 -->|Classical| P6[All Features]
        P5 -->|Quantum| P7[PCA/Dim Reduction]
    end

    subgraph "Hybrid Model Layer"
        P6 --> M1[XGBoost]
        P6 --> M2[Isolation Forest]
        
        P7 --> Q1[Quantum Feature Map (ZZ/Z)]
        Q1 --> Q2[Quantum Kernel (Fidelity)]
        Q1 --> Q3[VQC / QNN]
        Q3 --> O1[SPSA/COBYLA Optimizer]
        
        subgraph "Quantum Backend Abstraction"
            Q2 -.-> B1[Qiskit Simulator]
            Q3 -.-> B1
            Q2 -.-> B2[IBM Quantum Hardware]
            Q3 -.-> B2
            B2 --> R1[Qiskit Runtime Primitives (V2)]
        end
    end

    subgraph Evaluation
        M1 --> E1[ROC-AUC & F1]
        Q2 --> E1
        Q3 --> E1
        E1 --> Rpt[Performance Report]
    end

    subgraph Production
        M1 --> API[FastAPI Inference Service]
        Q3 --> API
        API --> UI[Streamlit Dashboard]
    end
```

## Design Decisions
1.  **Hybrid Approach**: We use XGBoost for high-throughput patterns and Quantum Models for capturing complex, non-linear relationships in high-value, ambiguous transactions.
2.  **Latency Handling**: Quantum models are computationally expensive. The system is designed to route only specific transactions to the QPU (Quantum Processing Unit), or use a pre-computed Quantum Kernel Support Vector Machine (QSVM).
3.  **Data Privacy**: No PII is sent to the quantum computer; only projected feature vectors (PCA components).
