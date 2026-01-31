# Quantum Explainability & Advantage

## Why Quantum Machine Learning (QML)?
In this project, we utilize **Quantum Kernel Methods** and **Variational Quantum Classifiers (VQC)** to detect fraud. The primary motivation is not just "speed" (which requires large fault-tolerant computers), but **expressibility** and **feature space separability** in the NISQ (Noisy Intermediate-Scale Quantum) era.

## 1. The Kernel Trick in Hilbert Space
Classical SVMs use kernels (like RBF) to map data into a higher-dimensional space where classes might be linearly separable.
$$ K(x, y) = \exp(-\gamma ||x - y||^2) $$

Quantum computers naturally operate in an exponentially large vector space ($2^N$ dimensions for $N$ qubits). We use a **Quantum Feature Map** $\phi(x)$ to map classical data $x$ into a quantum state $|\phi(x)\rangle$.
The Quantum Kernel is defined as:
$$ K_Q(x, y) = |\langle \phi(x) | \phi(y) \rangle|^2 $$

### Visual Proof
The heatmap generated in the dashboard (`src/explainability.py`) demonstrates this.
- **Classical RBF**: Often shows smooth, distance-based gradients.
- **Quantum Kernel**: Can reveal complex, periodic, or "blocky" structures based on the feature map's entanglement (ZZ-interactions). If the fraud patterns match this structure, the Quantum model will outperform the classical one.

## 2. Feature Map Circuit (`ZZFeatureMap`)
We use a **Second-Order Pauli-Z Evolution** feature map.
- **Input**: Scaled transaction features (PCA-reduced to 4 dimensions).
- **Entanglement**: Linear (Qubit 0 interacts with 1, 1 with 2, etc.).
- **Effect**: Creates complex correlations between features that classical models might miss without manual feature engineering.

## 3. Variational Quantum Classifier (VQC)
The VQC is a quantum neural network.
1.  **State Prep**: Encode data $|\phi(x)\rangle$.
2.  **Ansatz**: Apply trainable rotation layers $W(\theta)$ (TwoLocal).
3.  **Measurement**: Measure expectation value (Parity or Z-basis).
4.  **Label**: Map measurement to class 0 or 1.

## 4. Empirical Advantage
In our hybrid pipeline:
- **XGBoost**: Handles the "bulk" fraud patterns (rules, simple thresholds).
- **Quantum VQC**: Acts as a specialist for "subtle" patterns in the high-dimensional residue that XGBoost finds ambiguous.
