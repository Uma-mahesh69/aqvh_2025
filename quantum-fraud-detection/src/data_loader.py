import os
from typing import Tuple
import pandas as pd


def load_csvs(transaction_path: str, identity_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load transaction and identity CSVs.

    Args:
        transaction_path: Path to train_transaction.csv
        identity_path: Path to train_identity.csv

    Returns:
        Tuple of (df_transaction, df_identity)
    """
    if not os.path.exists(transaction_path):
        raise FileNotFoundError(f"Transaction CSV not found: {transaction_path}")
    if not os.path.exists(identity_path):
        raise FileNotFoundError(f"Identity CSV not found: {identity_path}")

    df_transaction = pd.read_csv(transaction_path)
    df_identity = pd.read_csv(identity_path)
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
