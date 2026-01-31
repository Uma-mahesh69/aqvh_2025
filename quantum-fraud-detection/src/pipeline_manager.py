"""
Pipeline Manager for Quantum Fraud Detection.
Handles end-to-end flow: Loading -> Preprocessing (w/ SMOTE) -> Training -> Evaluation.
"""
from __future__ import annotations
import os
import yaml
import logging
import time
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')

from src.data_loader import load_csvs, merge_on_transaction_id
from src.preprocessing import PreprocessConfig, preprocess_pipeline, split_data, apply_smote
from src.evaluation import compute_metrics, save_confusion_matrix, save_roc_curve, save_pr_curve
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
        os.makedirs(self.figures_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)

    def _load_config(self):
        with open(self.config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

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
        logging.info("Preprocessing Data...")
        pp_cfg = PreprocessConfig(
            missing_threshold=self.cfg["preprocessing"]["missing_threshold"],
            target_col=self.cfg["preprocessing"]["target_col"],
            feature_selection_method=self.cfg["preprocessing"].get("feature_selection_method", "pca"),
            top_k_features=self.cfg["preprocessing"].get("top_k_features", 8),
            use_llm_features=self.cfg["preprocessing"].get("use_llm_features", False),
            use_smote=self.cfg["preprocessing"].get("use_smote", False),
            smote_k_neighbors=self.cfg["preprocessing"].get("smote_k_neighbors", 5)
        )
        
        df_processed, selected_feats, artifacts = preprocess_pipeline(df, pp_cfg)
        self.artifacts.update(artifacts)
        
        # Save artifacts
        joblib.dump(artifacts, os.path.join(self.cfg["paths"]["results_dir"], "artifacts", "preprocess_artifacts.joblib"))
        
        # Split
        X_train, X_test, y_train, y_test = split_data(df_processed, pp_cfg.target_col)

        # Drop metadata from features
        cols_to_drop = ['TransactionID', 'TransactionDT', 'Transaction_datetime']
        X_train = X_train.drop(columns=[c for c in cols_to_drop if c in X_train.columns], errors='ignore')
        X_test = X_test.drop(columns=[c for c in cols_to_drop if c in X_test.columns], errors='ignore')
        
        # Apply SMOTE if requested (only on training data)
        if pp_cfg.use_smote:
            logging.info("Applying SMOTE to training data...")
            X_train, y_train = apply_smote(X_train, y_train, k_neighbors=pp_cfg.smote_k_neighbors)
            
        return X_train, X_test, y_train, y_test

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
                
            metrics = compute_metrics(y_test, y_pred, y_proba)
            metrics["training_time"] = duration
            logging.info(f"{name} Results: {metrics}")
            
            self.results[name] = metrics
            self.models[name] = model
            
            # Save Model
            safe_name = name.lower().replace(" ", "_")
            joblib.dump(model, os.path.join(self.results_dir, "models", f"{safe_name}.joblib"))
            
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
            shots=c.get("shots", 1024)
        )
        return QuantumConfig(
            num_features=self.cfg["preprocessing"]["top_k_features"],
            reps_feature_map=c["reps_feature_map"],
            reps_ansatz=c["reps_ansatz"],
            optimizer_maxiter=c["optimizer_maxiter"],
            backend_config=backend_cfg
        )

    def get_kernel_config(self):
        c = self.cfg["quantum_kernel"]
        q_cfg = self.cfg.get("quantum_backend", {})
        backend_cfg = BackendConfig(
            backend_type=q_cfg.get("backend_type", "simulator"),
            ibm_token=q_cfg.get("ibm_token"),
            ibm_backend_name=q_cfg.get("ibm_backend_name"),
            shots=c.get("shots", 1024)
        )
        return QuantumKernelConfig(
            num_features=self.cfg["preprocessing"]["top_k_features"], 
            backend_config=backend_cfg
        )
