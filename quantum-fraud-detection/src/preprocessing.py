from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split


@dataclass
class PreprocessConfig:
    missing_threshold: float = 50.0  # percent
    target_col: str = "isFraud"
    id_cols: Optional[List[str]] = None  # e.g., ["TransactionID"]
    top_k_corr_features: Optional[int] = 8  # None to disable corr-based selection


def drop_high_missing(df: pd.DataFrame, threshold_pct: float) -> pd.DataFrame:
    """Drop columns with missing values above threshold.
    
    Args:
        df: Input dataframe
        threshold_pct: Percentage threshold (0-100)
        
    Returns:
        Dataframe with high-missing columns removed
    """
    missing_pct = df.isnull().sum() / len(df) * 100
    cols_to_drop = missing_pct[missing_pct > threshold_pct].index.tolist()
    return df.drop(columns=cols_to_drop)


def impute_simple(df: pd.DataFrame) -> pd.DataFrame:
    """Simple imputation: median for numeric, mode for categorical.
    
    Args:
        df: Input dataframe
        
    Returns:
        Dataframe with imputed values
    """
    df = df.copy()
    for col in df.columns:
        if df[col].isnull().any():
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].median())
            else:
                mode_val = df[col].mode(dropna=True)
                df[col] = df[col].fillna(mode_val.iloc[0] if not mode_val.empty else "")
    return df


def label_encode_inplace(df: pd.DataFrame) -> pd.DataFrame:
    """Label encode all object-type columns.
    
    Args:
        df: Input dataframe
        
    Returns:
        Dataframe with encoded categorical columns
    """
    df = df.copy()
    for col in df.select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
    return df


def scale_numeric(df: pd.DataFrame, exclude: Optional[List[str]] = None) -> Tuple[pd.DataFrame, MinMaxScaler]:
    """Scale numeric columns using MinMaxScaler.
    
    Args:
        df: Input dataframe
        exclude: Columns to exclude from scaling
        
    Returns:
        Tuple of (scaled dataframe, fitted scaler)
    """
    exclude = exclude or []
    df = df.copy()
    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude]
    scaler = MinMaxScaler()
    if num_cols:
        df[num_cols] = scaler.fit_transform(df[num_cols])
    return df, scaler


def select_top_k_by_corr(df: pd.DataFrame, target: str, k: int) -> List[str]:
    """Select top k features by absolute correlation with target.
    
    Args:
        df: Input dataframe
        target: Target column name
        k: Number of features to select
        
    Returns:
        List of selected feature names
    """
    if target not in df.columns:
        raise KeyError(f"Target '{target}' not in dataframe")
    corr = df.corr(numeric_only=True)[target].drop(labels=[target])
    top = corr.abs().sort_values(ascending=False).head(k).index.tolist()
    return top


def split_data(
    df: pd.DataFrame,
    target: str,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split data into train and test sets.
    
    Args:
        df: Input dataframe
        target: Target column name
        test_size: Proportion of test set
        random_state: Random seed
        stratify: Whether to stratify split by target
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    X = df.drop(columns=[target])
    y = df[target]
    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if stratify else None,
    )


def preprocess_pipeline(
    df_merged: pd.DataFrame,
    cfg: PreprocessConfig,
) -> Tuple[pd.DataFrame, List[str]]:
    """Complete preprocessing pipeline.
    
    Args:
        df_merged: Merged input dataframe
        cfg: Preprocessing configuration
        
    Returns:
        Tuple of (processed dataframe, selected feature names)
    """
    df = drop_high_missing(df_merged, cfg.missing_threshold)
    df = impute_simple(df)
    df = label_encode_inplace(df)

    exclude = []
    if cfg.id_cols:
        exclude.extend(cfg.id_cols)
    exclude.append(cfg.target_col)
    df, _ = scale_numeric(df, exclude=exclude)

    selected_features: List[str]
    if cfg.top_k_corr_features and cfg.top_k_corr_features > 0:
        selected_features = select_top_k_by_corr(df, cfg.target_col, cfg.top_k_corr_features)
        keep_cols = selected_features + [cfg.target_col]
        # Add id_cols only if they exist in dataframe
        if cfg.id_cols:
            keep_cols.extend([col for col in cfg.id_cols if col in df.columns])
        df = df[[col for col in keep_cols if col in df.columns]]
    else:
        selected_features = [c for c in df.columns if c not in (cfg.id_cols or []) + [cfg.target_col]]

    return df, selected_features
