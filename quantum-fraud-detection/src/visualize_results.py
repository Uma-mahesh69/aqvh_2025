import json
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_ibm_results(json_path):
    print(f"Loading results from: {json_path}")
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    job_id = data.get("job_id", "unknown")
    backend = data.get("backend", "unknown")
    
    # Extract expectation values
    # Assuming single pub for VQC training
    if not data["data"]:
        print("No data found in JSON.")
        return

    evs = data["data"][0]["expectation_values"]
    iterations = range(1, len(evs) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, evs, 'o-', color='#6a0dad', label='Objective Function (Hardware)')
    
    plt.title(f"Quantum VQC Training Convergence on {backend}\nJob: {job_id}", fontsize=14)
    plt.xlabel("Optimization Step", fontsize=12)
    plt.ylabel("Expectation Value (Loss Proxy)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    output_path = f"results/figures/ibm_vqc_convergence_{job_id}.png"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    print(f"Plot saved to: {output_path}")

if __name__ == "__main__":
    # Automatically find the latest json in results
    import glob
    json_files = glob.glob("results/ibm_job_*.json")
    if json_files:
        latest_file = max(json_files, key=os.path.getctime)
        plot_ibm_results(latest_file)
    else:
        print("No result JSON files found.")
