"""
Quantum Explainability Module.
Generates visualizations to explain WHY Quantum Machine Learning is effective.
Focus: Feature Space Separability via Kernel Matrix comparison.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import rbf_kernel, cosine_similarity
from qiskit.circuit.library import ZZFeatureMap
from qiskit_algorithms.state_fidelities import ComputeUncompute
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit.primitives import StatevectorSampler as Sampler

def compute_and_plot_kernels(X_sample, y_sample, output_path="results/figures/kernel_comparison.png"):
    """
    Computes Quantum Kernel and Classical RBF Kernel for a sample.
    Plots side-by-side heatmaps to show difference in feature space structure.
    
    Args:
        X_sample: Small subset of data (e.g., 20-50 rows), scaled.
        y_sample: Labels for sorting/visualizing structure.
        output_path: Where to save the figure.
    """
    print("--- Generating Quantum Explainability Artifacts ---")
    
    # Sort data by class to make block structure visible
    sort_idx = np.argsort(y_sample)
    X_sorted = X_sample[sort_idx]
    
    # 1. Classical Kernel (RBF)
    print("Computing Classical RBF Kernel...")
    gamma = 1.0 / X_sorted.shape[1]
    K_classical = rbf_kernel(X_sorted, gamma=gamma)
    
    # 2. Quantum Kernel (ZZFeatureMap)
    print("Computing Quantum Kernel (ZZFeatureMap)...")
    num_features = X_sorted.shape[1]
    feature_map = ZZFeatureMap(feature_dimension=num_features, reps=2, entanglement='linear')
    sampler = Sampler() # Local simulator (Statevector)
    fidelity = ComputeUncompute(sampler=sampler)
    q_kernel = FidelityQuantumKernel(fidelity=fidelity, feature_map=feature_map)
    
    K_quantum = q_kernel.evaluate(X_sorted)
    
    # 3. Plotting
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Plot Classical
    sns.heatmap(K_classical, ax=axes[0], cmap="viridis", square=True)
    axes[0].set_title("Classical RBF Kernel\n(Feature Space Similarity)", fontsize=14)
    axes[0].set_xlabel("Sample Index (Sorted by Class)")
    axes[0].set_ylabel("Sample Index (Sorted by Class)")
    
    # Plot Quantum
    sns.heatmap(K_quantum, ax=axes[1], cmap="magma", square=True)
    axes[1].set_title("Quantum Kernel (Hilbert Space)\n(Projected Similarity)", fontsize=14)
    axes[1].set_xlabel("Sample Index (Sorted by Class)")
    axes[1].set_yticks([])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Kernel comparison saved to: {output_path}")
    
    return K_quantum

if __name__ == "__main__":
    # Test run
    # Mock data
    X = np.random.rand(20, 4)
    y = np.array([0]*10 + [1]*10)
    import os
    os.makedirs("results/figures", exist_ok=True)
    compute_and_plot_kernels(X, y)
