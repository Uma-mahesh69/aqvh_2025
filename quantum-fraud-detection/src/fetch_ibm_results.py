"""
Script to fetch and verify results from IBM Quantum Runtime.
Usage: python src/fetch_ibm_results.py
"""
from qiskit_ibm_runtime import QiskitRuntimeService
import numpy as np
import logging
import os
import yaml

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def retrieve_job(job_id):
    """Retrieve and inspect job results from IBM Quantum."""
    logging.info(f"Attempting to retrieve Job ID: {job_id}")
    
    # Load config to get token/instance if needed
    try:
        with open("configs/config.yaml", "r") as f:
            cfg = yaml.safe_load(f)
            ibm_token = cfg["quantum_backend"]["ibm_token"]
            
        # Fallback to env var if config token is empty or null
        if not ibm_token:
             logging.info("Token not found in config. Checking environment variable IBM_QUANTUM_TOKEN...")
             ibm_token = os.getenv("IBM_QUANTUM_TOKEN")
             
        # EMERGENCY FALBACK: Hardcode token from user input if all else fails
        if not ibm_token:
            logging.info("Using hardcoded fallback token.")
            ibm_token = "BorwkFjl-h1TPnKClBJQegN7emvIiEzB6kdftx60r3Zr"
             
    except Exception as e:
        logging.warning(f"Could not load config: {e}")
        ibm_token = os.getenv("IBM_QUANTUM_TOKEN")

    try:
        # Initialize Service
        if ibm_token:
            service = None
            # Attempt 1: Standard 'ibm_quantum'
            try:
                logging.info("Trying auth with channel='ibm_quantum'...")
                service = QiskitRuntimeService(channel="ibm_quantum", token=ibm_token)
            except Exception as e:
                 logging.info(f"Auth attempt 1 failed: {e}")
            
            # Attempt 2: User suggested 'ibm_quantum_platform' (newer qiskit versions?)
            if not service:
                try:
                    logging.info("Trying auth with channel='ibm_quantum_platform'...")
                    service = QiskitRuntimeService(channel="ibm_quantum_platform", token=ibm_token)
                except Exception as e:
                     logging.info(f"Auth attempt 2 failed: {e}")

            # Attempt 3: No channel (Inferred)
            if not service:
                try:
                    logging.info("Trying auth with token only (no channel arg)...")
                    service = QiskitRuntimeService(token=ibm_token)
                except Exception as e:
                     logging.error(f"All auth attempts failed. Last error: {e}")
                     return

        else:
            service = QiskitRuntimeService()
            
        # Get Job
        job = service.job(job_id)
        logging.info(f"Job Status: {job.status()}")
        logging.info(f"Job Backend: {job.backend().name}")
        
        # Get Result
        result = job.result()
        logging.info("Result retrieved successfully.")
        
        # Inspect Result Structure
        # Check if it's VQC (Estimator -> evs) or Kernel (Sampler -> quasi_dists)
        is_vqc = False
        
        # Primitive Result handling (V2)
        # Usually a list of PubResult
        if isinstance(result, list) or hasattr(result, '__iter__'):
             for idx, pub_res in enumerate(result):
                 logging.info(f"--- Pub {idx} ---")
                 if hasattr(pub_res, 'data'):
                     data = pub_res.data
                     if hasattr(data, 'evs'):
                         logging.info(f"Found Expectation Values (VQC): shape={data.evs.shape}")
                         is_vqc = True
                     elif hasattr(data, 'meas'): 
                         logging.info(f"Found Measurements (Sampler/Kernel): shape={data.meas.shape}")
                     elif hasattr(data, 'c'): # Post-processed counts sometimes
                          logging.info(f"Found Counts: {data.c}")
                     else:
                         logging.info(f"Data attributes: {dir(data)}")
                 else:
                     logging.info(f"Raw result: {pub_res}")

        if is_vqc:
            logging.info("Identified as VQC Job. Saving results locally...")
            import json
            output_data = {
                "job_id": job_id,
                "backend": job.backend().name,
                "status": job.status(),
                "data": []
            }
            
            # Extract data from pubs
            if isinstance(result, list) or hasattr(result, '__iter__'):
                for idx, pub_res in enumerate(result):
                    if hasattr(pub_res, 'data') and hasattr(pub_res.data, 'evs'):
                         # Convert numpy array to list for JSON serialization
                         evs = pub_res.data.evs.tolist() if hasattr(pub_res.data.evs, 'tolist') else list(pub_res.data.evs)
                         output_data["data"].append({"pub_idx": idx, "expectation_values": evs})
            
            output_file = f"results/ibm_job_{job_id}.json"
            with open(output_file, "w") as f:
                json.dump(output_data, f, indent=4)
            
            logging.info(f"✅ Results saved to: {output_file}")
            print(f"\nSUCCESS: Quantum Hardware Results saved to '{output_file}'")
            print("You can use this file to plot your VQC training convergence graph.")
            
    except Exception as e:
        logging.error(f"Failed to retrieve job: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Job ID from user screenshot/request
    JOB_ID = 'd5dlpq1smlfc739og8e0' 
    retrieve_job(JOB_ID)
