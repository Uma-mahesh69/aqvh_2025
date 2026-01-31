"""
Entry point for the SOTA Quantum Fraud Detection Pipeline.
Wraps the modular `FraudDetectionPipeline` class.
"""
import argparse
import sys
import logging
from src.pipeline_manager import FraudDetectionPipeline

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    parser = argparse.ArgumentParser(description="Run Quantum Fraud Detection Pipeline")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to configuration file")
    args = parser.parse_args()
    
    try:
        logging.info(f"Initializing Pipeline with config: {args.config}")
        pipeline = FraudDetectionPipeline(args.config)
        pipeline.run()
        logging.info("Pipeline finished successfully.")
    except Exception as e:
        logging.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
