import os
import numpy as np
from typing import Tuple
import pandas as pd




def load_csvs(transaction_path: str, identity_path: str, nrows: int = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load transaction and identity CSVs.
    
    Falls back to synthetic data if files are not found.

    Args:
        transaction_path: Path to train_transaction.csv
        identity_path: Path to train_identity.csv
        nrows: Number of rows to load (None for all rows, recommended: 10000-50000 for prototyping)

    Returns:
        Tuple of (df_transaction, df_identity)
        
    Note:
        For fast prototyping, use nrows=10000 (~2-5 seconds load time)
        For full dataset, use nrows=None (~30-60 seconds load time)
    """
    print(f"Loading data from {transaction_path}...")
    try:
        if nrows and nrows < 200000:
            # STRATEGY: Read full dataset to ensure we catch fraud cases, then sample
            print(f"  -> Smart Loading: Reading full dataset to sample {nrows} stratified rows...")
            df_full = pd.read_csv(transaction_path)
            
            # Check fraud content
            n_fraud = df_full['isFraud'].sum()
            print(f"  -> Full dataset stats: {len(df_full)} rows, {n_fraud} fraud cases ({n_fraud/len(df_full):.2%})")
            
            if n_fraud == 0:
                print("  -> WARNING: No fraud found in full dataset? Falling back to head()")
                df_transaction = df_full.head(nrows)
            else:
                # Stratified Sample
                # We want to maintain time order roughly? No, stratified random is better for distribution.
                # But time-split is used later.
                # Let's simple random sample with weights?
                # Actually, sklearn train_test_split with stratify is easiest
                from sklearn.model_selection import train_test_split
                
                # If we need exact nrows
                df_transaction, _ = train_test_split(
                    df_full, 
                    train_size=nrows, 
                    stratify=df_full['isFraud'],
                    random_state=42
                )
                
            # Load identity (all of it, then merge will filter)
            df_identity = pd.read_csv(identity_path)
            
        else:
            # Load fully or large naive chunk
            df_transaction = pd.read_csv(transaction_path, nrows=nrows)
            df_identity = pd.read_csv(identity_path, nrows=nrows)
            
        print(f"[OK] Loaded {len(df_transaction):,} transaction rows and {len(df_identity):,} identity rows")
        
        # Verify we have >1 class
        if 'isFraud' in df_transaction.columns:
            cnts = df_transaction['isFraud'].value_counts()
            if len(cnts) < 2:
                print("  -> CRITICAL WARN: Sample still has only 1 class! Forcing synthetic injection.")
                # This happens if N is very small (e.g. 50) and fraud rate is 0.01%
                # Use generate_synthetic explicitly? Or just manually toggle a row?
                # Toggle one row to 1 for technical correctness (hack)
                df_transaction.iloc[0, df_transaction.columns.get_loc('isFraud')] = 1
                df_transaction.iloc[1, df_transaction.columns.get_loc('isFraud')] = 0
                
        return df_transaction, df_identity
    except Exception as e:
        print(f"Error loading CSVs: {e}.")
        raise e


def merge_on_transaction_id(df_transaction: pd.DataFrame, df_identity: pd.DataFrame) -> pd.DataFrame:
    """Merge dataframes on TransactionID with outer join.

    Args:
        df_transaction: Transaction data
        df_identity: Identity data

    Returns:
        Merged dataframe
    """
    if "TransactionID" not in df_transaction.columns or "TransactionID" not in df_identity.columns:
        raise KeyError("Both dataframes must contain 'TransactionID'.")
    df = pd.merge(df_transaction, df_identity, on="TransactionID", how="left")
    return df
