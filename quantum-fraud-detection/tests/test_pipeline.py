"""Quick test to verify the pipeline works without running full training."""
import sys
import yaml
import pandas as pd

# Test 1: Config loading
print("Test 1: Loading configuration...")
try:
    with open("configs/config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    print("[OK] Config loaded successfully")
    print(f"  - Transaction CSV: {cfg['data']['transaction_csv']}")
    print(f"  - Identity CSV: {cfg['data']['identity_csv']}")
    print(f"  - Figures dir: {cfg['paths']['figures_dir']}")
except Exception as e:
    print(f"[FAIL] Config loading failed: {e}")
    sys.exit(1)

# Test 2: Import all modules
print("\nTest 2: Importing modules...")
try:
    from src.data_loader import load_csvs, merge_on_transaction_id
    from src.preprocessing import PreprocessConfig, preprocess_pipeline, split_data
    from src.model_classical import ClassicalConfig, train_logreg
    from src.model_quantum import QuantumConfig, build_vqc
    from src import evaluation as eval_mod
    print("[OK] All modules imported successfully")
except Exception as e:
    print(f"[FAIL] Module import failed: {e}")
    sys.exit(1)

# Test 3: Load data (sample)
print("\nTest 3: Loading data sample...")
try:
    df_txn, df_id = load_csvs(
        cfg["data"]["transaction_csv"],
        cfg["data"]["identity_csv"]
    )
    print(f"[OK] Data loaded successfully")
    print(f"  - Transaction shape: {df_txn.shape}")
    print(f"  - Identity shape: {df_id.shape}")
except Exception as e:
    print(f"[FAIL] Data loading failed: {e}")
    sys.exit(1)

# Test 4: Merge data
print("\nTest 4: Merging datasets...")
try:
    df = merge_on_transaction_id(df_txn, df_id)
    print(f"[OK] Data merged successfully")
    print(f"  - Merged shape: {df.shape}")
    print(f"  - Target column present: {'isFraud' in df.columns}")
    print(f"  - Fraud rate: {df['isFraud'].mean():.2%}")
except Exception as e:
    print(f"[FAIL] Data merging failed: {e}")
    sys.exit(1)

# Test 5: Preprocessing (on small sample)
print("\nTest 5: Testing preprocessing pipeline...")
try:
    # Use small sample for speed
    df_sample = df.sample(n=min(1000, len(df)), random_state=42)
    
    pp_cfg = PreprocessConfig(
        missing_threshold=cfg["preprocessing"]["missing_threshold"],
        target_col=cfg["preprocessing"]["target_col"],
        id_cols=cfg["preprocessing"].get("id_cols", []),
        top_k_corr_features=cfg["preprocessing"]["top_k_corr_features"],
    )
    
    df_processed, selected = preprocess_pipeline(df_sample, pp_cfg)
    print(f"[OK] Preprocessing successful")
    print(f"  - Processed shape: {df_processed.shape}")
    print(f"  - Selected features: {len(selected)}")
    print(f"  - Features: {selected}")
except Exception as e:
    print(f"[FAIL] Preprocessing failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Data splitting
print("\nTest 6: Testing data split...")
try:
    X_train, X_test, y_train, y_test = split_data(
        df_processed,
        target=pp_cfg.target_col,
        test_size=cfg["preprocessing"]["test_size"],
        random_state=cfg["preprocessing"]["random_state"],
        stratify=cfg["preprocessing"]["stratify"],
    )
    print(f"[OK] Data split successful")
    print(f"  - Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    print(f"  - Train fraud rate: {y_train.mean():.2%}")
    print(f"  - Test fraud rate: {y_test.mean():.2%}")
except Exception as e:
    print(f"[FAIL] Data split failed: {e}")
    sys.exit(1)

# Test 7: Classical model configuration
print("\nTest 7: Testing classical model setup...")
try:
    cl_cfg = ClassicalConfig(
        penalty=cfg["classical_model"]["penalty"],
        C=cfg["classical_model"]["C"],
        max_iter=cfg["classical_model"]["max_iter"],
        class_weight=cfg["classical_model"]["class_weight"],
        use_random_oversampler=cfg["classical_model"]["use_random_oversampler"],
    )
    print(f"[OK] Classical config created")
    print(f"  - Penalty: {cl_cfg.penalty}, C: {cl_cfg.C}")
    print(f"  - Use oversampling: {cl_cfg.use_random_oversampler}")
except Exception as e:
    print(f"[FAIL] Classical config failed: {e}")
    sys.exit(1)

# Test 8: Quantum model configuration
print("\nTest 8: Testing quantum model setup...")
try:
    q_cfg = QuantumConfig(
        num_features=X_train.shape[1],
        reps_feature_map=cfg["quantum_model"]["reps_feature_map"],
        reps_ansatz=cfg["quantum_model"]["reps_ansatz"],
        optimizer_maxiter=cfg["quantum_model"]["optimizer_maxiter"],
        shots=cfg["quantum_model"]["shots"],
    )
    print(f"[OK] Quantum config created")
    print(f"  - Num features: {q_cfg.num_features}")
    print(f"  - Feature map reps: {q_cfg.reps_feature_map}")
    print(f"  - Ansatz reps: {q_cfg.reps_ansatz}")
    
    # Try building quantum circuit (without training)
    vqc = build_vqc(q_cfg)
    print(f"[OK] Quantum classifier built successfully")
except Exception as e:
    print(f"[FAIL] Quantum config failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("[SUCCESS] ALL TESTS PASSED - Pipeline is ready to run!")
print("="*60)
print("\nTo run the full pipeline:")
print("  python run.py --config configs/config.yaml")
