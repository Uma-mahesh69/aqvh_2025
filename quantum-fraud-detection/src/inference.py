"""
Inference Engine for Quantum Fraud Detection.
Handles loading models, artifacts, and processing single transactions with context.
"""
import os
import joblib
import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Tuple

import sys
sys.path.append(os.getcwd())

from src.preprocessing import engineer_features, drop_high_missing, impute_simple, label_encode_inplace
# We need to import the exact classes used in pickling if they are custom
from src.preprocessing import PreprocessConfig  # Ensure this is imported if needed for pickle
from src.feature_llm import LLMFeatureExtractor # Needed for unpickling

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class FraudInference:
    def __init__(self, artifacts_path: str, models_dir: str, background_data_path: str = None):
        """
        Initialize Inference Engine.
        
        Args:
            artifacts_path: Path to preprocess_artifacts.joblib
            models_dir: Directory containing trained models
            background_data_path: Path to a CSV to use as 'context' for feature engineering.
        """
        self.artifacts = self._load_artifacts(artifacts_path)
        self.models = self._load_models(models_dir)
        self.thresholds = self._load_thresholds(models_dir)
        self.background_df = self._load_background_data(background_data_path)
        
    def _load_artifacts(self, path: str) -> Dict:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Artifacts not found at {path}")
        logging.info(f"Loading artifacts from {path}...")
        return joblib.load(path)
        
    def _load_models(self, models_dir: str) -> Dict:
        models = {}
        # Load XGBoost
        xgb_path = os.path.join(models_dir, "xgboost.joblib")
        if os.path.exists(xgb_path):
            logging.info("Loading XGBoost model...")
            models['xgboost'] = joblib.load(xgb_path)
            
        # Load Quantum VQC
        vqc_path = os.path.join(models_dir, "quantum_vqc.joblib")
        if os.path.exists(vqc_path):
            logging.info("Loading Quantum VQC model...")
            models['quantum_vqc'] = joblib.load(vqc_path)
            
        return models

    def _load_thresholds(self, models_dir: str) -> Dict:
        thresholds = {}
        for model_name in ['xgboost', 'quantum_vqc']:
            path = os.path.join(models_dir, f"{model_name}_threshold.json")
            if os.path.exists(path):
                try:
                    import json
                    with open(path, "r") as f:
                        data = json.load(f)
                        thresholds[model_name] = data.get("threshold", 0.5)
                        logging.info(f"Loaded optimal threshold for {model_name}: {thresholds[model_name]}")
                except:
                    thresholds[model_name] = 0.5
            else:
                thresholds[model_name] = 0.5
        return thresholds

    def _load_background_data(self, path: str) -> pd.DataFrame:
        """Load a small subset of training data to support rolling features/aggregations."""
        if path and os.path.exists(path):
            logging.info(f"Loading background context from {path}...")
            # Load small sample (e.g., 500 rows) to keep inference fast but contextual
            df = pd.read_csv(path, nrows=500) 
            return df
        return pd.DataFrame()

    def preprocess_single(self, transaction_data: Dict[str, Any]) -> pd.DataFrame:
        """
        Preprocess a single transaction dictionary.
        Strategy: Append to background data -> Process -> Extract last row.
        """
        # Convert dict to DataFrame (1 row)
        df_single = pd.DataFrame([transaction_data])
        
        # Combine with background for context (aggregations, frequencies)
        if not self.background_df.empty:
            # Ensure columns match, fill missing in single with NaN
            # (In production, we'd have a strict schema)
           
            # Concatenate
            df_combined = pd.concat([self.background_df, df_single], axis=0, ignore_index=True)
            target_idx = len(df_combined) - 1
        else:
            df_combined = df_single
            target_idx = 0
            
        # --- 1. Feature Engineering ---
        # (Re-using exactly the same function as training)
        df_eng = engineer_features(df_combined)
        
        # --- 1.5 LLM Features ---
        if 'llm_extractor' in self.artifacts:
            extractor = self.artifacts['llm_extractor']
            # We assume extractor is already fitted
            text_cols = ['TransactionAmt', 'ProductCD', 'card4', 'card6', 'P_emaildomain', 'R_emaildomain']
            df_llm = extractor.transform(df_eng, text_cols)
            df_eng = pd.concat([df_eng, df_llm], axis=1)

        # --- 2. Cleaning (Simplified) ---
        # We assume background data is already cleaned/imputed if it comes from raw. 
        # But actually we loaded raw CSV. Ideally we should run impute_simple.
        df_eng = impute_simple(df_eng)

        # --- 3. Encoding ---
        # Convert strings to numbers. 
        # CAUTION: Since we fit fresh LabelEncoders, the mapping might slightly differ from training 
        # if background_df doesn't cover all categories. Accepted risk for Hackathon demo.
        df_eng = label_encode_inplace(df_eng)

        # --- 4. Transformation (Pipeline) ---
        # The pipeline handles Scaling and PCA
        
        if 'pipeline' in self.artifacts:
            pipeline = self.artifacts['pipeline']
            input_feats = self.artifacts['input_features']
            
            # Ensure all input features exist (fill 0 if missing in this single row context due to label encoding differences)
            for f in input_feats:
                if f not in df_eng.columns:
                    df_eng[f] = 0
            
            # Extract just the target row
            # We want a DataFrame with the exact columns expected by the pipeline
            X_input = df_eng.iloc[[target_idx]][input_feats]
            
            # Validating input integrity (fill NaNs one last time as pipeline expects numeric)
            X_input = X_input.fillna(0)
            
            # Transform using the fitted pipeline
            # This applies StandardScaler and PCA (if configured) exactly as in training
            X_transformed = pipeline.transform(X_input)
            
            # Create DataFrame with proper column names
            output_feats = self.artifacts.get('output_features', [f"Feature_{i}" for i in range(X_transformed.shape[1])])
            df_final = pd.DataFrame(X_transformed, columns=output_feats)
            
        else:
            # Legacy fallback (should not be hit in new flow)
            logging.warning("Pipeline not found in artifacts. Using legacy column selection.")
            selected_feats = self.artifacts.get('selected_features', [])
            for f in selected_feats:
                if f not in df_eng.columns:
                    df_eng[f] = 0
            df_final = df_eng.iloc[[target_idx]][selected_feats].reset_index(drop=True)
            
        return df_final

    def predict(self, transaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run inference on a single transaction.
        """
        try:
            X = self.preprocess_single(transaction_data)
            
            results = {}
            results['features'] = X.values.tolist()[0] # For 3D viz
            
            # XGBoost Prediction
            if 'xgboost' in self.models:
                xgb = self.models['xgboost']
                prob = xgb.predict_proba(X)[0][1]
                threshold = self.thresholds.get('xgboost', 0.5)
                
                results['xgboost_prob'] = float(prob)
                results['xgboost_pred'] = int(prob >= threshold)
                results['xgboost_threshold'] = threshold
            
            # Quantum Prediction
            if 'quantum_vqc' in self.models:
                vqc = self.models['quantum_vqc']
                threshold = self.thresholds.get('quantum_vqc', 0.5)
                # Qiskit classifier predict might return class, predict_proba returns probs
                try:
                    q_prob = vqc.predict_proba(X.values)[0][1]
                    results['quantum_prob'] = float(q_prob)
                    results['quantum_pred'] = int(q_prob >= threshold)
                except:
                    # Fallback if predict_proba fails (some QNN versions)
                    q_pred = vqc.predict(X.values)[0]
                    results['quantum_prob'] = float(q_pred) # 0.0 or 1.0
                    results['quantum_pred'] = int(q_pred) # Already binary?
                    
                results['quantum_threshold'] = threshold
                    
            return results
            
        except Exception as e:
            logging.error(f"Prediction failed: {e}", exc_info=True)
            return {"error": str(e)}

if __name__ == "__main__":
    # Test
    print("Testing Inference Engine...")
    engine = FraudInference(
        artifacts_path="results/artifacts/preprocess_artifacts.joblib",
        models_dir="results/models",
        background_data_path="data/train_transaction.csv"
    )
    
    # Mock Input
    sample = {
        'TransactionAmt': 150.0,
        'ProductCD': 'W',
        'card4': 'visa',
        'card6': 'debit',
        'P_emaildomain': 'gmail.com',
        'TransactionDT': 86400, # 1 day
        'card1': 1000,
        'card2': 555,
        'addr1': 300
    }
    
    res = engine.predict(sample)
    print("Prediction Result:", res)
