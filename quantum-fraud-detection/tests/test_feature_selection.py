"""
Quick test script for advanced feature selection methods.
"""
import sys
import yaml
import logging
from src.data_loader import load_csvs, merge_on_transaction_id
from src.preprocessing import PreprocessConfig, preprocess_pipeline

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

print("="*80)
print("TESTING ADVANCED FEATURE SELECTION")
print("="*80)

# Load config
with open("configs/config.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

# Load small sample for testing
print("\n[1/4] Loading data (1000 rows for quick test)...")
try:
    df_txn, df_id = load_csvs(
        cfg["data"]["transaction_csv"],
        cfg["data"]["identity_csv"],
        nrows=1000
    )
    df = merge_on_transaction_id(df_txn, df_id)
    print(f"[OK] Loaded {len(df)} merged records")
except Exception as e:
    print(f"[FAIL] Failed to load data: {e}")
    sys.exit(1)

# Test each feature selection method
methods = ["correlation", "mutual_info", "rf_importance", "rfe", "ensemble"]

for method in methods:
    print(f"\n[Testing] Method: {method}")
    print("-" * 60)
    
    try:
        pp_cfg = PreprocessConfig(
            missing_threshold=cfg["preprocessing"]["missing_threshold"],
            target_col=cfg["preprocessing"]["target_col"],
            id_cols=cfg["preprocessing"].get("id_cols", []),
            feature_selection_method=method,
            top_k_features=8,
            ensemble_voting_threshold=2,
        )
        
        df_processed, selected = preprocess_pipeline(df, pp_cfg)
        
        print(f"[OK] {method.upper()} completed successfully")
        print(f"  Selected {len(selected)} features: {selected}")
        print(f"  Final shape: {df_processed.shape}")
        
    except Exception as e:
        print(f"[FAIL] {method.upper()} failed: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "="*80)
print("FEATURE SELECTION TEST COMPLETED")
print("="*80)
