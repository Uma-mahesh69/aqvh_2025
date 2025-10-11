import os
import argparse
import yaml
import numpy as np
import logging

from src.data_loader import load_csvs, merge_on_transaction_id
from src.preprocessing import PreprocessConfig, preprocess_pipeline, split_data
from src.model_classical import ClassicalConfig, train_logreg
from src.model_quantum import QuantumConfig, train_vqc
from src import evaluation as eval_mod

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def main(config_path: str):
    """Run end-to-end fraud detection pipeline with classical and quantum models.
    
    Args:
        config_path: Path to YAML configuration file
    """
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Paths
    transaction_csv = cfg["data"]["transaction_csv"]
    identity_csv = cfg["data"]["identity_csv"]
    figures_dir = cfg["paths"]["figures_dir"]
    
    # Ensure output directories exist
    os.makedirs(figures_dir, exist_ok=True)
    logging.info(f"Output directory: {figures_dir}")

    # Load + merge
    df_txn, df_id = load_csvs(transaction_csv, identity_csv)
    df = merge_on_transaction_id(df_txn, df_id)

    # Preprocess
    pp_cfg = PreprocessConfig(
        missing_threshold=cfg["preprocessing"]["missing_threshold"],
        target_col=cfg["preprocessing"]["target_col"],
        id_cols=cfg["preprocessing"].get("id_cols", []),
        top_k_corr_features=cfg["preprocessing"]["top_k_corr_features"],
    )
    df_processed, selected = preprocess_pipeline(df, pp_cfg)

    # Split
    X_train, X_test, y_train, y_test = split_data(
        df_processed,
        target=pp_cfg.target_col,
        test_size=cfg["preprocessing"]["test_size"],
        random_state=cfg["preprocessing"]["random_state"],
        stratify=cfg["preprocessing"]["stratify"],
    )

    # Train classical
    cl_cfg = ClassicalConfig(
        penalty=cfg["classical_model"]["penalty"],
        C=cfg["classical_model"]["C"],
        max_iter=cfg["classical_model"]["max_iter"],
        class_weight=cfg["classical_model"]["class_weight"],
        use_random_oversampler=cfg["classical_model"]["use_random_oversampler"],
    )
    clf_classical = train_logreg(X_train, y_train, cl_cfg)

    # Evaluate classical
    y_pred_cl = clf_classical.predict(X_test)
    y_proba_cl = None
    try:
        y_proba_cl = clf_classical.predict_proba(X_test)[:, 1]
    except Exception as e:
        logging.warning(f"Could not get probability predictions for classical model: {e}")
    metrics_cl = eval_mod.compute_metrics(y_test, y_pred_cl, y_proba_cl)
    eval_mod.save_confusion_matrix(y_test, y_pred_cl, os.path.join(figures_dir, "confusion_classical.png"))
    eval_mod.save_roc_curve(clf_classical, X_test, y_test, os.path.join(figures_dir, "roc_classical.png"))

    # Train quantum (using selected features only)
    q_num_features = X_train.shape[1]
    q_cfg = QuantumConfig(
        num_features=q_num_features,
        reps_feature_map=cfg["quantum_model"]["reps_feature_map"],
        reps_ansatz=cfg["quantum_model"]["reps_ansatz"],
        optimizer_maxiter=cfg["quantum_model"]["optimizer_maxiter"],
        shots=cfg["quantum_model"]["shots"],
    )
    try:
        clf_quantum = train_vqc(X_train.values if hasattr(X_train, "values") else X_train,
                                y_train.values if hasattr(y_train, "values") else y_train,
                                q_cfg)
        y_pred_q = clf_quantum.predict(X_test.values if hasattr(X_test, "values") else X_test)
        y_proba_q = None
        try:
            y_proba_q = clf_quantum.predict_proba(X_test.values if hasattr(X_test, "values") else X_test)[:, 1]
        except Exception as e:
            logging.warning(f"Could not get probability predictions for quantum model: {e}")
        metrics_q = eval_mod.compute_metrics(y_test, y_pred_q, y_proba_q)
        eval_mod.save_confusion_matrix(y_test, y_pred_q, os.path.join(figures_dir, "confusion_quantum.png"))
        # ROC may not be directly supported; attempt via predict_proba
        if y_proba_q is not None:
            eval_mod.save_roc_curve(clf_quantum, X_test, y_test, os.path.join(figures_dir, "roc_quantum.png"))
    except Exception as e:
        logging.error(f"Quantum training skipped due to error: {e}", exc_info=True)
        metrics_q = None

    logging.info(f"Classical metrics: {metrics_cl}")
    if metrics_q is not None:
        logging.info(f"Quantum metrics: {metrics_q}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args = parser.parse_args()
    main(args.config)
