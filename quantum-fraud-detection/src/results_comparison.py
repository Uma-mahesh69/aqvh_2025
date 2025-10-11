"""Results comparison and visualization for classical vs quantum models."""
from __future__ import annotations
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json


def save_metrics_comparison(
    metrics_dict: Dict[str, Dict[str, float]],
    output_path: str,
    title: str = "Model Performance Comparison"
) -> None:
    """Save comparison of metrics across models.
    
    Args:
        metrics_dict: Dictionary mapping model names to their metrics
        output_path: Path to save the comparison figure
        title: Title for the plot
    """
    # Convert to DataFrame
    df = pd.DataFrame(metrics_dict).T
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1']
    
    for idx, metric in enumerate(metrics_to_plot):
        ax = axes[idx // 2, idx % 2]
        
        if metric in df.columns:
            # Sort by metric value
            sorted_df = df.sort_values(by=metric, ascending=False)
            
            # Create bar plot
            bars = ax.bar(range(len(sorted_df)), sorted_df[metric])
            
            # Color bars differently for classical vs quantum
            for i, (model_name, _) in enumerate(sorted_df.iterrows()):
                if 'quantum' in model_name.lower() or 'vqc' in model_name.lower() or 'kernel' in model_name.lower():
                    bars[i].set_color('#FF6B6B')  # Red for quantum
                else:
                    bars[i].set_color('#4ECDC4')  # Teal for classical
            
            ax.set_xticks(range(len(sorted_df)))
            ax.set_xticklabels(sorted_df.index, rotation=45, ha='right')
            ax.set_ylabel(metric.capitalize())
            ax.set_title(f'{metric.capitalize()} Comparison')
            ax.grid(axis='y', alpha=0.3)
            
            # Add value labels on bars
            for i, v in enumerate(sorted_df[metric]):
                ax.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def save_metrics_table(
    metrics_dict: Dict[str, Dict[str, float]],
    output_path: str
) -> None:
    """Save metrics as a formatted table.
    
    Args:
        metrics_dict: Dictionary mapping model names to their metrics
        output_path: Path to save the table (CSV format)
    """
    df = pd.DataFrame(metrics_dict).T
    df = df.round(4)
    
    # Sort by F1 score
    if 'f1' in df.columns:
        df = df.sort_values(by='f1', ascending=False)
    
    df.to_csv(output_path)
    print(f"Metrics table saved to: {output_path}")


def save_training_time_comparison(
    time_dict: Dict[str, float],
    output_path: str,
    title: str = "Training Time Comparison"
) -> None:
    """Save comparison of training times across models.
    
    Args:
        time_dict: Dictionary mapping model names to training time in seconds
        output_path: Path to save the comparison figure
        title: Title for the plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Sort by time
    sorted_items = sorted(time_dict.items(), key=lambda x: x[1])
    models = [item[0] for item in sorted_items]
    times = [item[1] for item in sorted_items]
    
    # Create bar plot
    bars = ax.barh(range(len(models)), times)
    
    # Color bars differently for classical vs quantum
    for i, model_name in enumerate(models):
        if 'quantum' in model_name.lower() or 'vqc' in model_name.lower() or 'kernel' in model_name.lower():
            bars[i].set_color('#FF6B6B')  # Red for quantum
        else:
            bars[i].set_color('#4ECDC4')  # Teal for classical
    
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models)
    ax.set_xlabel('Training Time (seconds)')
    ax.set_title(title, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, v in enumerate(times):
        ax.text(v + max(times) * 0.01, i, f'{v:.2f}s', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def create_quantum_advantage_report(
    metrics_dict: Dict[str, Dict[str, float]],
    time_dict: Dict[str, float],
    output_path: str
) -> None:
    """Create a comprehensive report comparing quantum vs classical models.
    
    Args:
        metrics_dict: Dictionary mapping model names to their metrics
        time_dict: Dictionary mapping model names to training time
        output_path: Path to save the report (text file)
    """
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("QUANTUM vs CLASSICAL FRAUD DETECTION - COMPREHENSIVE REPORT")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    # Separate classical and quantum models
    classical_models = {k: v for k, v in metrics_dict.items() 
                       if not any(x in k.lower() for x in ['quantum', 'vqc', 'kernel'])}
    quantum_models = {k: v for k, v in metrics_dict.items() 
                     if any(x in k.lower() for x in ['quantum', 'vqc', 'kernel'])}
    
    # Classical models summary
    report_lines.append("CLASSICAL MODELS PERFORMANCE")
    report_lines.append("-" * 80)
    for model_name, metrics in classical_models.items():
        report_lines.append(f"\n{model_name}:")
        for metric, value in metrics.items():
            report_lines.append(f"  {metric.capitalize()}: {value:.4f}")
        if model_name in time_dict:
            report_lines.append(f"  Training Time: {time_dict[model_name]:.2f}s")
    
    report_lines.append("\n")
    report_lines.append("QUANTUM MODELS PERFORMANCE")
    report_lines.append("-" * 80)
    for model_name, metrics in quantum_models.items():
        report_lines.append(f"\n{model_name}:")
        for metric, value in metrics.items():
            report_lines.append(f"  {metric.capitalize()}: {value:.4f}")
        if model_name in time_dict:
            report_lines.append(f"  Training Time: {time_dict[model_name]:.2f}s")
    
    # Best model analysis
    report_lines.append("\n")
    report_lines.append("BEST MODELS BY METRIC")
    report_lines.append("-" * 80)
    
    df = pd.DataFrame(metrics_dict).T
    for metric in ['accuracy', 'precision', 'recall', 'f1']:
        if metric in df.columns:
            best_model = df[metric].idxmax()
            best_value = df[metric].max()
            model_type = "Quantum" if any(x in best_model.lower() for x in ['quantum', 'vqc', 'kernel']) else "Classical"
            report_lines.append(f"{metric.capitalize()}: {best_model} ({model_type}) - {best_value:.4f}")
    
    # Quantum advantage analysis
    report_lines.append("\n")
    report_lines.append("QUANTUM ADVANTAGE ANALYSIS")
    report_lines.append("-" * 80)
    
    if classical_models and quantum_models:
        classical_df = pd.DataFrame(classical_models).T
        quantum_df = pd.DataFrame(quantum_models).T
        
        for metric in ['accuracy', 'precision', 'recall', 'f1']:
            if metric in classical_df.columns and metric in quantum_df.columns:
                best_classical = classical_df[metric].max()
                best_quantum = quantum_df[metric].max()
                improvement = ((best_quantum - best_classical) / best_classical) * 100
                
                report_lines.append(f"\n{metric.capitalize()}:")
                report_lines.append(f"  Best Classical: {best_classical:.4f}")
                report_lines.append(f"  Best Quantum: {best_quantum:.4f}")
                report_lines.append(f"  Improvement: {improvement:+.2f}%")
                
                if improvement > 0:
                    report_lines.append(f"  → Quantum models show advantage!")
                elif improvement < -5:
                    report_lines.append(f"  → Classical models perform better")
                else:
                    report_lines.append(f"  → Performance is comparable")
    
    # Conclusion
    report_lines.append("\n")
    report_lines.append("CONCLUSION")
    report_lines.append("-" * 80)
    
    if quantum_models and classical_models:
        quantum_df = pd.DataFrame(quantum_models).T
        classical_df = pd.DataFrame(classical_models).T
        
        if 'f1' in quantum_df.columns and 'f1' in classical_df.columns:
            best_quantum_f1 = quantum_df['f1'].max()
            best_classical_f1 = classical_df['f1'].max()
            
            if best_quantum_f1 > best_classical_f1:
                report_lines.append("Quantum models demonstrate superior performance in fraud detection,")
                report_lines.append("showing the potential of quantum machine learning for this task.")
            elif best_quantum_f1 < best_classical_f1 * 0.95:
                report_lines.append("Classical models currently outperform quantum models, likely due to")
                report_lines.append("limited training data, noise, or optimization challenges in quantum circuits.")
            else:
                report_lines.append("Quantum and classical models show comparable performance,")
                report_lines.append("suggesting quantum models are competitive even in early stages.")
    
    report_lines.append("\n")
    report_lines.append("=" * 80)
    
    # Save report
    with open(output_path, 'w') as f:
        f.write('\n'.join(report_lines))
    
    print(f"Quantum advantage report saved to: {output_path}")


def save_all_results(
    metrics_dict: Dict[str, Dict[str, float]],
    time_dict: Dict[str, float],
    output_dir: str
) -> None:
    """Save all comparison results and visualizations.
    
    Args:
        metrics_dict: Dictionary mapping model names to their metrics
        time_dict: Dictionary mapping model names to training time
        output_dir: Directory to save all outputs
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save metrics comparison plot
    save_metrics_comparison(
        metrics_dict,
        str(output_path / "metrics_comparison.png"),
        title="Classical vs Quantum Models - Performance Comparison"
    )
    
    # Save metrics table
    save_metrics_table(
        metrics_dict,
        str(output_path / "metrics_table.csv")
    )
    
    # Save training time comparison
    if time_dict:
        save_training_time_comparison(
            time_dict,
            str(output_path / "training_time_comparison.png")
        )
    
    # Create quantum advantage report
    create_quantum_advantage_report(
        metrics_dict,
        time_dict,
        str(output_path / "quantum_advantage_report.txt")
    )
    
    # Save raw results as JSON
    results = {
        "metrics": metrics_dict,
        "training_times": time_dict
    }
    with open(output_path / "results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nAll results saved to: {output_dir}")
