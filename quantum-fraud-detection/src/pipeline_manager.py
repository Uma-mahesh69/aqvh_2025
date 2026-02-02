"""
Pipeline Manager for Quantum Fraud Detection.
Handles end-to-end flow: Loading -> Preprocessing (w/ SMOTE) -> Training -> Evaluation.
"""
from __future__ import annotations
import os
import yaml
import logging
import json
import time
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')

from src.data_loader import load_csvs, merge_on_transaction_id
from src.preprocessing import PreprocessConfig, preprocess_pipeline, split_data, apply_smote, create_transform_pipeline
from src.evaluation import compute_metrics, save_confusion_matrix, save_roc_curve, save_pr_curve, find_optimal_threshold
from src.model_classical import (
    ClassicalConfig, train_logreg,
    IsolationForestConfig, train_isolation_forest,
    XGBoostConfig, train_xgboost
)
from src.model_quantum import (
    QuantumConfig, train_vqc,
    QuantumKernelConfig, train_quantum_kernel
)
from src.quantum_backend import BackendConfig
from src.config_schema import validate_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fraud_pipeline.log'),
        logging.StreamHandler()
    ]
)

class FraudDetectionPipeline:
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.cfg = self._load_config()
        self.artifacts = {}
        self.models = {}
        self.results = {}
        
        # Paths
        self.figures_dir = self.cfg["paths"]["figures_dir"]
        self.results_dir = self.cfg["paths"]["results_dir"]
        self.artifacts_dir = os.path.join(self.results_dir, "artifacts")
        self.models_dir = os.path.join(self.results_dir, "models")
        os.makedirs(self.figures_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.artifacts_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)

    def _load_config(self):
        with open(self.config_path, "r", encoding="utf-8") as f:
            raw_data = yaml.safe_load(f)
        
        # Validate schema and return as dict to preserve existing access patterns
        validated_config = validate_config(raw_data)
        logging.info("Configuration Schema Validated Successfully.")
        return validated_config.model_dump()

    def run(self):
        """Execute the full pipeline."""
        df = self.load_data()
        X_train, X_test, y_train, y_test = self.preprocess_data(df)
        
        # Models to run config
        run_cfg = self.cfg.get("models_to_run", {})

        # Train & Evaluate Models
        if run_cfg.get("logistic_regression", True):
            self.train_evaluate("Logistic Regression", train_logreg, self.get_classical_config(), X_train, y_train, X_test, y_test)
        
        if run_cfg.get("isolation_forest", True):
            self.train_evaluate("Isolation Forest", train_isolation_forest, self.get_iso_forest_config(), X_train, y_train, X_test, y_test)
        
        if run_cfg.get("xgboost", True):
            self.train_evaluate("XGBoost", train_xgboost, self.get_xgboost_config(), X_train, y_train, X_test, y_test)
        
        # Quantum Models
        if run_cfg.get("quantum_vqc", False):
            self.train_evaluate("Quantum VQC", train_vqc, self.get_quantum_config(), X_train, y_train, X_test, y_test)
            
        if run_cfg.get("quantum_kernel", False):
            self.train_evaluate("Quantum Kernel", train_quantum_kernel, self.get_kernel_config(), X_train, y_train, X_test, y_test)
            
        # Save overarching results
        self.save_summary()

    def load_data(self):
        logging.info("Loading Data...")
        nrows = self.cfg["data"].get("nrows")
        df_txn, df_id = load_csvs(self.cfg["data"]["transaction_csv"], self.cfg["data"]["identity_csv"], nrows=nrows)
        return merge_on_transaction_id(df_txn, df_id)

    def preprocess_data(self, df):
        logging.info("Preprocessing Data (Leakage-Free Flow)...")
        pp_cfg = PreprocessConfig(
            missing_threshold=self.cfg["preprocessing"]["missing_threshold"],
            target_col=self.cfg["data"]["target_col"],
            feature_selection_method=self.cfg["preprocessing"].get("feature_selection_method", "pca"),
            top_k_features=self.cfg["preprocessing"].get("top_k_features", 8),
            use_llm_features=self.cfg["preprocessing"].get("use_llm_features", False),
            use_smote=self.cfg["preprocessing"].get("use_smote", False),
            smote_k_neighbors=self.cfg["preprocessing"].get("smote_k_neighbors", 5)
        )
        
        # 1. Feature Engineering & Cleaning (Row-wise only)
        # Returns df with all features, unscaled.
        df_processed, input_features, artifacts = preprocess_pipeline(df, pp_cfg)
        
        # 2. Split Data FIRST (Crucial for preventing leakage)
        # We split the ID/Time columns too so we can drop them later
        X_train_df, X_test_df, y_train, y_test = split_data(df_processed, pp_cfg.target_col)
        
        logging.info(f"Data Split: Train={len(X_train_df)}, Test={len(X_test_df)}")
        
        # 3. Create & Fit Transformation Pipeline (Scaler -> PCA)
        # Fit ONLY on Training Data
        transform_pipeline, output_feature_names = create_transform_pipeline(pp_cfg, len(input_features))
        
        logging.info("Fitting Transformation Pipeline on Training Data...")
        # Select only the input feature columns
        X_train_input = X_train_df[input_features]
        transform_pipeline.fit(X_train_input)
        
        # 4. Transform Data
        X_train_transformed = transform_pipeline.transform(X_train_input)
        X_test_transformed = transform_pipeline.transform(X_test_df[input_features])
        
        # Store pipeline for inference
        artifacts['pipeline'] = transform_pipeline
        artifacts['input_features'] = input_features # Features expected by pipeline
        artifacts['output_features'] = output_feature_names # Features produced by pipeline (e.g. PCA_0...)
        
        self.artifacts.update(artifacts)
        joblib.dump(artifacts, os.path.join(self.artifacts_dir, "preprocess_artifacts.joblib"))
        
        # 5. Apply SMOTE (Optional, Train data only)
        if pp_cfg.use_smote:
            logging.info("Applying SMOTE to training data...")
            X_train_final, y_train_final = apply_smote(X_train_transformed, y_train, k_neighbors=pp_cfg.smote_k_neighbors)
        else:
            X_train_final, y_train_final = X_train_transformed, y_train
            
        return X_train_final, X_test_transformed, y_train_final, y_test

    def train_evaluate(self, name, train_func, config, X_train, y_train, X_test, y_test):
        logging.info(f"--- Training {name} ---")
        try:
            start = time.time()
            model = train_func(X_train, y_train, config)
            duration = time.time() - start
            
            # Predict
            y_pred = model.predict(X_test)
            # Fix Isolation Forest predictions
            if name == "Isolation Forest":
                y_pred = np.where(y_pred == -1, 1, 0)
                
            y_proba = None
            if hasattr(model, "predict_proba"):
                try:
                    y_proba = model.predict_proba(X_test)[:, 1]
                except:
                    pass
            elif hasattr(model, "decision_function"):
                y_proba = model.decision_function(X_test)
                # Normalize decision function to 0-1 for common thresholding logic via sigmoid
                # Or just use raw score if compute_metrics handles it (but find_optimal expects proba-like)
                # For simplicity in hackathon, we skip optimal threshold for SVM if it returns raw distance
                # Or apply sigmoid:
                y_proba = 1 / (1 + np.exp(-y_proba))
                
            # Optimal Thresholding
            optimal_thresh = 0.5
            if y_proba is not None:
                optimal_thresh, best_f1 = find_optimal_threshold(y_test, y_proba)
                logging.info(f"{name} Optimal Threshold: {optimal_thresh:.4f} (Max F1: {best_f1:.4f})")
                
                # Re-compute predictions based on optimal threshold
                y_pred = (y_proba >= optimal_thresh).astype(int)

            metrics = compute_metrics(y_test, y_pred, y_proba)
            metrics["training_time"] = duration
            metrics["optimal_threshold"] = optimal_thresh
            logging.info(f"{name} Results: {metrics}")
            
            self.results[name] = metrics
            self.models[name] = model
            
            # Save Model
            safe_name = name.lower().replace(" ", "_")
            joblib.dump(model, os.path.join(self.models_dir, f"{safe_name}.joblib"))
            
            # Save Threshold
            thresh_path = os.path.join(self.models_dir, f"{safe_name}_threshold.json")
            with open(thresh_path, "w") as f:
                json.dump({"threshold": optimal_thresh, "model": name}, f)
            
            # Visualization
            save_confusion_matrix(y_test, y_pred, os.path.join(self.figures_dir, f"confusion_{safe_name}.png"))
            if y_proba is not None:
                save_roc_curve(model, X_test, y_test, os.path.join(self.figures_dir, f"roc_{safe_name}.png"))
                save_pr_curve(model, X_test, y_test, os.path.join(self.figures_dir, f"pr_curve_{safe_name}.png"))
                
        except Exception as e:
            logging.error(f"Failed to train {name}: {e}", exc_info=True)

    def save_summary(self):
        df_res = pd.DataFrame(self.results).T
        df_res.to_csv(os.path.join(self.results_dir, "metrics_table.csv"))
        logging.info("Pipeline Execution Complete. Summary saved.")
        
        # Generate Quantum Advantage Report
        self.generate_report()

    def generate_report(self):
        """Generate a text report comparing classical and quantum models."""
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("QUANTUM vs CLASSICAL FRAUD DETECTION - COMPREHENSIVE REPORT")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        # Separate classical and quantum models
        classical_models = {k: v for k, v in self.results.items() 
                           if not any(x in k.lower() for x in ['quantum', 'vqc', 'kernel'])}
        quantum_models = {k: v for k, v in self.results.items() 
                         if any(x in k.lower() for x in ['quantum', 'vqc', 'kernel'])}
        
        # Classical models summary
        report_lines.append("CLASSICAL MODELS PERFORMANCE")
        report_lines.append("-" * 80)
        for model_name, metrics in classical_models.items():
            report_lines.append(f"\n{model_name}:")
            for metric, value in metrics.items():
                if isinstance(value, (int, float)):
                    report_lines.append(f"  {metric.capitalize()}: {value:.4f}")
        
        report_lines.append("\n")
        report_lines.append("QUANTUM MODELS PERFORMANCE")
        report_lines.append("-" * 80)
        for model_name, metrics in quantum_models.items():
            report_lines.append(f"\n{model_name}:")
            for metric, value in metrics.items():
                if isinstance(value, (int, float)):
                    report_lines.append(f"  {metric.capitalize()}: {value:.4f}")
        
        # Conclusion
        report_lines.append("\n")
        report_lines.append("=" * 80)
        
        out_path = os.path.join(self.results_dir, "quantum_advantage_report.txt")
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        logging.info(f"Report saved to {out_path}")

    # Configuration Helpers
    def get_classical_config(self):
        c = self.cfg["logistic_regression"]
        return ClassicalConfig(penalty=c["penalty"], C=c["C"], class_weight=c["class_weight"])
        
    def get_iso_forest_config(self):
        c = self.cfg["isolation_forest"]
        return IsolationForestConfig(n_estimators=c["n_estimators"], contamination=c["contamination"])
        
    def get_xgboost_config(self):
        c = self.cfg["xgboost"]
        return XGBoostConfig(n_estimators=c["n_estimators"], max_depth=c["max_depth"], use_gpu=c["use_gpu"])

    def get_quantum_config(self):
        c = self.cfg["quantum_vqc"]
        q_cfg = self.cfg.get("quantum_backend", {})
        backend_cfg = BackendConfig(
            backend_type=q_cfg.get("backend_type", "simulator"),
            ibm_token=q_cfg.get("ibm_token"),
            ibm_backend_name=q_cfg.get("ibm_backend_name"),
            shots=c.get("shots", 1024),
            optimization_level=q_cfg.get("optimization_level", 3),
            resilience_level=q_cfg.get("resilience_level", 1)
        )
        
        # Use actual output features if available (post-pipeline), else config default
        num_feats = len(self.artifacts.get("output_features", [])) 
        if num_feats == 0:
            num_feats = self.cfg["preprocessing"].get("top_k_features", 8)
            
        return QuantumConfig(
            num_features=num_feats,
            reps_feature_map=c.get("reps_feature_map", 2),
            reps_ansatz=c.get("reps_ansatz", 2),
            optimizer_maxiter=c.get("optimizer_maxiter", 100),
            shots=c.get("shots", 1024),
            optimizer=c.get("optimizer", "cobyla"),
            backend_config=backend_cfg
        )

    def get_kernel_config(self):
        c = self.cfg["quantum_kernel"]
        q_cfg = self.cfg.get("quantum_backend", {})
        backend_cfg = BackendConfig(
            backend_type=q_cfg.get("backend_type", "simulator"),
            ibm_token=q_cfg.get("ibm_token"),
            ibm_backend_name=q_cfg.get("ibm_backend_name"),
            shots=c.get("shots", 1024),
            optimization_level=q_cfg.get("optimization_level", 3),
            resilience_level=q_cfg.get("resilience_level", 1)
        )
        
        # Use actual output features if available
        num_feats = len(self.artifacts.get("output_features", [])) 
        if num_feats == 0:
            num_feats = self.cfg["preprocessing"].get("top_k_features", 8)

        return QuantumKernelConfig(
            num_features=num_feats,
            reps_feature_map=c.get("reps_feature_map", 2),
            shots=c.get("shots", 1024),
            C=c.get("C", 1.0),
            gamma=c.get("gamma", "scale"),
            backend_config=backend_cfg
        )
