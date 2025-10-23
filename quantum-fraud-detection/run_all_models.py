"""
Comprehensive fraud detection pipeline comparing classical and quantum models.

This script runs:
- Classical Models: Logistic Regression, Isolation Forest, XGBoost
- Quantum Models: VQC, Quantum Kernel
- Backends: Simulator, Aer, IBM Quantum Hardware
"""
import os
import argparse
import yaml
import numpy as np
import logging
import time
from pathlib import Path

# Configure matplotlib to use non-interactive backend (prevents tkinter threading warnings)
import matplotlib
matplotlib.use('Agg')

from src.data_loader import load_csvs, merge_on_transaction_id
from src.preprocessing import PreprocessConfig, preprocess_pipeline, split_data, split_data_time_based
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
from src import evaluation as eval_mod
from src.results_comparison import save_all_results

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fraud_detection_pipeline.log'),
        logging.StreamHandler()
    ]
)


def train_and_evaluate_model(
    model_name: str,
    train_func,
    config,
    X_train,
    y_train,
    X_test,
    y_test,
    figures_dir: str
):
    """Train and evaluate a single model.
    
    Args:
        model_name: Name of the model
        train_func: Training function
        config: Model configuration
        X_train, y_train: Training data
        X_test, y_test: Test data
        figures_dir: Directory to save figures
        
    Returns:
        Tuple of (metrics_dict, training_time, model)
    """
    logging.info(f"\n{'='*80}")
    logging.info(f"Training {model_name}...")
    logging.info(f"{'='*80}")
    
    start_time = time.time()
    
    try:
        # Convert to numpy if needed
        X_train_np = X_train.values if hasattr(X_train, "values") else X_train
        y_train_np = y_train.values if hasattr(y_train, "values") else y_train
        X_test_np = X_test.values if hasattr(X_test, "values") else X_test
        y_test_np = y_test.values if hasattr(y_test, "values") else y_test
        
        # Train model
        model = train_func(X_train_np, y_train_np, config)
        training_time = time.time() - start_time
        
        logging.info(f"Training completed in {training_time:.2f} seconds")
        
        # Predict
        y_pred = model.predict(X_test_np)
        
        # Handle Isolation Forest predictions (-1, 1) -> (0, 1)
        if model_name == "Isolation Forest":
            y_pred = np.where(y_pred == -1, 1, 0)  # -1 (anomaly) -> 1 (fraud)
        
        # Get probabilities if available
        y_proba = None
        try:
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X_test_np)[:, 1]
            elif hasattr(model, 'decision_function'):
                y_proba = model.decision_function(X_test_np)
        except Exception as e:
            logging.warning(f"Could not get probability predictions: {e}")
        
        # Compute metrics
        metrics = eval_mod.compute_metrics(y_test_np, y_pred, y_proba)
        
        # Save visualizations
        safe_name = model_name.replace(" ", "_").lower()
        eval_mod.save_confusion_matrix(
            y_test_np, y_pred,
            os.path.join(figures_dir, f"confusion_{safe_name}.png")
        )
        
        if y_proba is not None:
            try:
                eval_mod.save_roc_curve(
                    model, X_test, y_test,
                    os.path.join(figures_dir, f"roc_{safe_name}.png")
                )
            except Exception as e:
                logging.warning(f"Could not save ROC curve: {e}")
        
        logging.info(f"{model_name} Metrics: {metrics}")
        
        return metrics, training_time, model
        
    except Exception as e:
        logging.error(f"Error training {model_name}: {e}", exc_info=True)
        return None, None, None


def main(config_path: str):
    """Run comprehensive fraud detection pipeline.
    
    Args:
        config_path: Path to YAML configuration file
    """
    logging.info("Starting Quantum Fraud Detection Pipeline")
    logging.info(f"Configuration file: {config_path}")
    
    # Load configuration
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    
    # Setup paths
    transaction_csv = cfg["data"]["transaction_csv"]
    identity_csv = cfg["data"]["identity_csv"]
    nrows = cfg["data"].get("nrows")  # Get nrows from config (None for all rows)
    figures_dir = cfg["paths"]["figures_dir"]
    results_dir = cfg["paths"]["results_dir"]
    
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # Load and preprocess data
    logging.info("\n" + "="*80)
    logging.info("Loading and preprocessing data...")
    if nrows:
        logging.info(f"PROTOTYPING MODE: Loading only {nrows:,} rows for faster testing")
    else:
        logging.info("FULL MODE: Loading all ~590,000 rows")
    logging.info("="*80)
    
    df_txn, df_id = load_csvs(transaction_csv, identity_csv, nrows=nrows)
    df = merge_on_transaction_id(df_txn, df_id)
    
    pp_cfg = PreprocessConfig(
        missing_threshold=cfg["preprocessing"]["missing_threshold"],
        target_col=cfg["preprocessing"]["target_col"],
        id_cols=cfg["preprocessing"].get("id_cols", []),
        top_k_corr_features=cfg["preprocessing"].get("top_k_corr_features"),  # Optional parameter
        feature_selection_method=cfg["preprocessing"].get("feature_selection_method", "ensemble"),
        top_k_features=cfg["preprocessing"].get("top_k_features", 8),
        ensemble_voting_threshold=cfg["preprocessing"].get("ensemble_voting_threshold", 2),
    )
    df_processed, selected = preprocess_pipeline(df, pp_cfg)
    
    logging.info(f"\n{'='*80}")
    logging.info(f"FEATURE SELECTION COMPLETE")
    logging.info(f"Method: {pp_cfg.feature_selection_method}")
    logging.info(f"Selected {len(selected)} features: {selected}")
    logging.info(f"{'='*80}\n")
    
    # Split data - Use time-based validation for fraud detection
    use_time_based = cfg["preprocessing"].get("use_time_based_split", True)
    
    if use_time_based:
        logging.info("Using TIME-BASED validation (prevents data leakage)")
        X_train, X_test, y_train, y_test = split_data_time_based(
            df_processed,
            target=pp_cfg.target_col,
            time_col='TransactionDT',
            test_size=cfg["preprocessing"]["test_size"],
        )
    else:
        logging.info("Using RANDOM validation (legacy mode)")
        X_train, X_test, y_train, y_test = split_data(
            df_processed,
            target=pp_cfg.target_col,
            test_size=cfg["preprocessing"]["test_size"],
            random_state=cfg["preprocessing"]["random_state"],
            stratify=cfg["preprocessing"]["stratify"],
        )
    
    logging.info(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    logging.info(f"Features: {X_train.shape[1]}")
    logging.info(f"Class distribution - Train: {np.bincount(y_train.astype(int))}, Test: {np.bincount(y_test.astype(int))}")
    
    # Storage for results
    all_metrics = {}
    all_times = {}
    models_to_run = cfg.get("models_to_run", {})
    
    # ========== CLASSICAL MODELS ==========
    
    # 1. Logistic Regression
    if models_to_run.get("logistic_regression", True):
        lr_cfg = ClassicalConfig(
            penalty=cfg["logistic_regression"]["penalty"],
            C=cfg["logistic_regression"]["C"],
            max_iter=cfg["logistic_regression"]["max_iter"],
            class_weight=cfg["logistic_regression"]["class_weight"],
            use_random_oversampler=cfg["logistic_regression"]["use_random_oversampler"],
        )
        metrics, train_time, model = train_and_evaluate_model(
            "Logistic Regression", train_logreg, lr_cfg,
            X_train, y_train, X_test, y_test, figures_dir
        )
        if metrics:
            all_metrics["Logistic Regression"] = metrics
            all_times["Logistic Regression"] = train_time
    
    # 2. Isolation Forest
    if models_to_run.get("isolation_forest", True):
        if_cfg = IsolationForestConfig(
            n_estimators=cfg["isolation_forest"]["n_estimators"],
            contamination=cfg["isolation_forest"]["contamination"],
            max_samples=cfg["isolation_forest"]["max_samples"],
            random_state=cfg["isolation_forest"]["random_state"],
        )
        metrics, train_time, model = train_and_evaluate_model(
            "Isolation Forest", train_isolation_forest, if_cfg,
            X_train, y_train, X_test, y_test, figures_dir
        )
        if metrics:
            all_metrics["Isolation Forest"] = metrics
            all_times["Isolation Forest"] = train_time
    
    # 3. XGBoost
    if models_to_run.get("xgboost", True):
        xgb_cfg = XGBoostConfig(
            n_estimators=cfg["xgboost"]["n_estimators"],
            max_depth=cfg["xgboost"]["max_depth"],
            learning_rate=cfg["xgboost"]["learning_rate"],
            subsample=cfg["xgboost"].get("subsample", 1.0),
            colsample_bytree=cfg["xgboost"].get("colsample_bytree", 1.0),
            scale_pos_weight=cfg["xgboost"]["scale_pos_weight"],
            random_state=cfg["xgboost"]["random_state"],
            use_gpu=cfg["xgboost"]["use_gpu"],
        )
        metrics, train_time, model = train_and_evaluate_model(
            "XGBoost", train_xgboost, xgb_cfg,
            X_train, y_train, X_test, y_test, figures_dir
        )
        if metrics:
            all_metrics["XGBoost"] = metrics
            all_times["XGBoost"] = train_time
    
    # ========== QUANTUM MODELS ==========
    
    # Setup backend configuration
    backend_cfg = BackendConfig(
        backend_type=cfg["quantum_backend"]["backend_type"],
        ibm_token=cfg["quantum_backend"].get("ibm_token") or os.getenv("IBM_QUANTUM_TOKEN"),
        ibm_backend_name=cfg["quantum_backend"].get("ibm_backend_name"),
        shots=cfg["quantum_backend"]["shots"],
        optimization_level=cfg["quantum_backend"]["optimization_level"],
    )
    
    num_features = X_train.shape[1]
    
    # 4. Variational Quantum Classifier (VQC)
    if models_to_run.get("quantum_vqc", True):
        vqc_cfg = QuantumConfig(
            num_features=num_features,
            reps_feature_map=cfg["quantum_vqc"]["reps_feature_map"],
            reps_ansatz=cfg["quantum_vqc"]["reps_ansatz"],
            optimizer_maxiter=cfg["quantum_vqc"]["optimizer_maxiter"],
            shots=cfg["quantum_vqc"]["shots"],
            backend_config=backend_cfg,
        )
        metrics, train_time, model = train_and_evaluate_model(
            "Quantum VQC", train_vqc, vqc_cfg,
            X_train, y_train, X_test, y_test, figures_dir
        )
        if metrics:
            all_metrics["Quantum VQC"] = metrics
            all_times["Quantum VQC"] = train_time
    
    # 5. Quantum Kernel
    if models_to_run.get("quantum_kernel", True):
        qk_cfg = QuantumKernelConfig(
            num_features=num_features,
            reps_feature_map=cfg["quantum_kernel"]["reps_feature_map"],
            shots=cfg["quantum_kernel"]["shots"],
            C=cfg["quantum_kernel"]["C"],
            gamma=cfg["quantum_kernel"]["gamma"],
            backend_config=backend_cfg,
        )
        metrics, train_time, model = train_and_evaluate_model(
            "Quantum Kernel", train_quantum_kernel, qk_cfg,
            X_train, y_train, X_test, y_test, figures_dir
        )
        if metrics:
            all_metrics["Quantum Kernel"] = metrics
            all_times["Quantum Kernel"] = train_time
    
    # ========== RESULTS COMPARISON ==========
    
    logging.info("\n" + "="*80)
    logging.info("Generating comprehensive results comparison...")
    logging.info("="*80)
    
    if all_metrics:
        save_all_results(all_metrics, all_times, results_dir)
        
        # Print summary
        logging.info("\n" + "="*80)
        logging.info("FINAL RESULTS SUMMARY")
        logging.info("="*80)
        
        for model_name, metrics in all_metrics.items():
            logging.info(f"\n{model_name}:")
            for metric, value in metrics.items():
                logging.info(f"  {metric}: {value:.4f}")
            if model_name in all_times:
                logging.info(f"  Training Time: {all_times[model_name]:.2f}s")
    else:
        logging.warning("No models were successfully trained!")
    
    logging.info("\n" + "="*80)
    logging.info("Pipeline completed successfully!")
    logging.info(f"Results saved to: {results_dir}")
    logging.info("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run comprehensive quantum fraud detection pipeline"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to YAML configuration file"
    )
    args = parser.parse_args()
    
    main(args.config)
