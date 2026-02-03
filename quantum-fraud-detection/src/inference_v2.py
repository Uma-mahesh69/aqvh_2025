"""
Inference Engine V2
Compatible with sklearn Pipeline artifacts from preprocessing_v2.
"""
import os
import joblib
import pandas as pd
import numpy as np
import logging
import json
from pathlib import Path
from typing import Dict, Any, Union

# Ensure custom transformers are valid during unpickling
from src.preprocessing_v2 import (
    FeatureEngineer, ColumnDropper, CategoricalEncoder, 
    DataFrameImputer, DataFrameScaler, DimensionalityReducer
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FraudInferenceV2:
    def __init__(self, artifacts_dir: str, models_dir: str):
        """
        Initialize V2 Inference Engine.
        
        Args:
            artifacts_dir: Directory containing pipeline.pkl
            models_dir: Directory containing trained models
        """
        self.artifacts_dir = Path(artifacts_dir)
        self.models_dir = Path(models_dir)
        
        # Load Pipeline (Find the latest or specific one)
        self.pipeline = self._load_latest_pipeline()
        
        # Load Models and Thresholds
        self.models = {}
        self.thresholds = {}
        self._load_models()
        
    def _load_latest_pipeline(self):
        """Find and load the latest pipeline.pkl in artifacts dir."""
        try:
            pipelines = list(self.artifacts_dir.glob("*_pipeline.pkl"))
            if not pipelines:
                raise FileNotFoundError("No pipeline artifacts found.")
            
            # Sort by modification time (newest first)
            latest_pipeline = sorted(pipelines, key=lambda x: x.stat().st_mtime, reverse=True)[0]
            logger.info(f"Loading pipeline from {latest_pipeline}")
            return joblib.load(latest_pipeline)
        
        except Exception as e:
            logger.error(f"Failed to load pipeline: {e}")
            raise

    def _load_models(self):
        """Load all available models and their thresholds."""
        model_files = list(self.models_dir.glob("*_*.pkl"))
        
        for p in model_files:
            # Filename format: {exp_id}_{model_name}.pkl
            # We want to extract {model_name}
            # Heuristic: split by '_' and take the last part if it's a known model name, 
            # or try to match known keys.
            
            fname = p.stem # e.g. "FraudExp_v2..._xgboost"
            
            if "xgboost" in fname:
                name = "xgboost"
            elif "quantum_vqc" in fname:
                name = "quantum_vqc"
            elif "logistic_regression" in fname:
                name = "logistic_regression"
            else:
                continue
                
            logger.info(f"Loading model: {name} from {p.name}")
            self.models[name] = joblib.load(p)
            
            # Try load threshold
            thresh_path = self.models_dir / f"{fname}_threshold.json"
            if thresh_path.exists():
                with open(thresh_path, 'r') as f:
                    data = json.load(f)
                    self.thresholds[name] = data.get('threshold', 0.5)
            else:
                self.thresholds[name] = 0.5

    def predict_single(self, transaction_data: Dict[str, Any], model_name: str = 'xgboost') -> Dict[str, Any]:
        """
        Predict fraud probability for a single transaction.
        
        Args:
            transaction_data: Dictionary of transaction features.
            model_name: Model to use for prediction.
            
        Returns:
            Dictionary with prediction results.
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not available. Available: {list(self.models.keys())}")
            
        try:
            # 1. Convert dict to DataFrame
            df = pd.DataFrame([transaction_data])
            
            # 2. Transform using Pipeline
            # Note: The pipeline expects certain columns. 
            # If logic requires historical context (background data) for aggressive feature engineering,
            # that should have been handled in the 'FeatureEngineer' transformer state or 
            # passed here. Currently V2 FeatureEngineer uses fitted self.freq_encodings_, 
            # so it is STATEFUL and independent of background data at inference time! (HUGE WIN)
            
            X_transformed = self.pipeline.transform(df)
            
            # 3. Predict
            model = self.models[model_name]
            
            # Check if model expects specific input format (e.g. DMatrix for native XGB)
            # The models saved in V2 seem to be wrapped or sklearn compatible.
            
            if hasattr(model, "predict_proba"):
                prob = model.predict_proba(X_transformed)[0, 1]
            else:
                # Fallback
                prob = float(model.predict(X_transformed)[0])
                
            threshold = self.thresholds.get(model_name, 0.5)
            is_fraud = prob >= threshold
            
            return {
                "fraud_probability": float(prob),
                "is_fraud": bool(is_fraud),
                "threshold_used": float(threshold),
                "model_used": model_name,
                "risk_level": "Critical" if prob > 0.8 else ("High" if prob > 0.5 else "Low")
            }
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}", exc_info=True)
            return {"error": str(e), "fraud_probability": 0.0, "is_fraud": False}
