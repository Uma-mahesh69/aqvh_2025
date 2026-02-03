
import os
import shutil
import glob
import logging

logging.basicConfig(level=logging.INFO)

def clean_results():
    # 1. Remove results_test directory
    if os.path.exists("results_test"):
        logging.info("Removing results_test directory...")
        shutil.rmtree("results_test")
        
    # 2. Remove stale root result files
    stale_files = [
        "results/results.json",
        "results/metrics_comparison.png", 
        "results/training_time_comparison.png"
    ]
    # Wildcard for ibm_job
    stale_files.extend(glob.glob("results/ibm_job_*.json"))
    
    for f in stale_files:
        if os.path.exists(f):
            logging.info(f"Removing {f}...")
            os.remove(f)
            
    # 3. Remove stale figures (Quantum Kernel was not run)
    stale_figures = glob.glob("results/figures/*quantum_kernel*")
    # Also old convergence plots with IDs
    stale_figures.extend(glob.glob("results/figures/ibm_vqc_convergence_*.png"))
    
    for f in stale_figures:
        if os.path.exists(f):
            logging.info(f"Removing {f}...")
            os.remove(f)

    # 4. Remove stale models (Quantum Kernel)
    stale_models = glob.glob("results/models/*quantum_kernel*")
    for f in stale_models:
        if os.path.exists(f):
            logging.info(f"Removing {f}...")
            os.remove(f)

    print("✅ Cleanup Complete.")

if __name__ == "__main__":
    clean_results()
