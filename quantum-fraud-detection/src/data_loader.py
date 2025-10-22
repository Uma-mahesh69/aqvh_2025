import os
from typing import Tuple
import pandas as pd


def load_csvs(transaction_path: str, identity_path: str, nrows: int = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load transaction and identity CSVs.

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
    if not os.path.exists(transaction_path):
        raise FileNotFoundError(f"Transaction CSV not found: {transaction_path}")
    if not os.path.exists(identity_path):
        raise FileNotFoundError(f"Identity CSV not found: {identity_path}")

    print(f"Loading data with nrows={nrows if nrows else 'all (~590k rows)'}...")
    df_transaction = pd.read_csv(transaction_path, nrows=nrows)
    df_identity = pd.read_csv(identity_path, nrows=nrows)
    print(f"[OK] Loaded {len(df_transaction):,} transaction rows and {len(df_identity):,} identity rows")
    return df_transaction, df_identity


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
    df = pd.merge(df_transaction, df_identity, on="TransactionID", how="outer")
    return df
