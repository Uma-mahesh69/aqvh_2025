from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from collections import Counter
import logging


@dataclass
class PreprocessConfig:
    missing_threshold: float = 50.0  # percent
    target_col: str = "isFraud"
    id_cols: Optional[List[str]] = None  # e.g., ["TransactionID"]
    top_k_corr_features: Optional[int] = 8  # None to disable corr-based selection
    # Advanced feature selection options
    feature_selection_method: str = "ensemble"  # "correlation", "mutual_info", "rf_importance", "rfe", "ensemble"
    top_k_features: Optional[int] = 8  # Number of features to select (overrides top_k_corr_features if set)
    ensemble_voting_threshold: int = 2  # Minimum votes needed from methods (for ensemble)


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


def select_by_mutual_info(df: pd.DataFrame, target: str, k: int, random_state: int = 42) -> List[str]:
    """Select top k features by mutual information with target.
    
    Mutual information captures non-linear relationships between features and target.
    
    Args:
        df: Input dataframe
        target: Target column name
        k: Number of features to select
        random_state: Random seed for reproducibility
        
    Returns:
        List of selected feature names
    """
    if target not in df.columns:
        raise KeyError(f"Target '{target}' not in dataframe")
    
    X = df.drop(columns=[target])
    y = df[target]
    
    # Compute mutual information scores
    mi_scores = mutual_info_classif(X, y, random_state=random_state)
    mi_scores = pd.Series(mi_scores, index=X.columns)
    
    # Select top k features
    top_features = mi_scores.sort_values(ascending=False).head(k).index.tolist()
    logging.info(f"Mutual Info - Top {k} features: {top_features}")
    
    return top_features


def select_by_rf_importance(df: pd.DataFrame, target: str, k: int, random_state: int = 42) -> List[str]:
    """Select top k features by Random Forest feature importance.
    
    Random Forest importance captures non-linear interactions and is robust to outliers.
    
    Args:
        df: Input dataframe
        target: Target column name
        k: Number of features to select
        random_state: Random seed for reproducibility
        
    Returns:
        List of selected feature names
    """
    if target not in df.columns:
        raise KeyError(f"Target '{target}' not in dataframe")
    
    X = df.drop(columns=[target])
    y = df[target]
    
    # Train Random Forest for feature importance
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=random_state, n_jobs=-1)
    rf.fit(X, y)
    
    # Get feature importances
    importances = pd.Series(rf.feature_importances_, index=X.columns)
    top_features = importances.sort_values(ascending=False).head(k).index.tolist()
    
    logging.info(f"RF Importance - Top {k} features: {top_features}")
    
    return top_features


def select_by_rfe(df: pd.DataFrame, target: str, k: int, random_state: int = 42) -> List[str]:
    """Select top k features using Recursive Feature Elimination (RFE).
    
    RFE iteratively removes the least important features based on model coefficients.
    
    Args:
        df: Input dataframe
        target: Target column name
        k: Number of features to select
        random_state: Random seed for reproducibility
        
    Returns:
        List of selected feature names
    """
    if target not in df.columns:
        raise KeyError(f"Target '{target}' not in dataframe")
    
    X = df.drop(columns=[target])
    y = df[target]
    
    # Use Logistic Regression as the estimator for RFE
    estimator = LogisticRegression(max_iter=1000, random_state=random_state)
    rfe = RFE(estimator=estimator, n_features_to_select=k, step=1)
    rfe.fit(X, y)
    
    # Get selected features
    top_features = X.columns[rfe.support_].tolist()
    
    logging.info(f"RFE - Top {k} features: {top_features}")
    
    return top_features


def select_by_ensemble(df: pd.DataFrame, target: str, k: int, voting_threshold: int = 2, random_state: int = 42) -> Tuple[List[str], Dict[str, int]]:
    """Select features using ensemble voting from multiple methods.
    
    Combines correlation, mutual information, RF importance, and RFE.
    Features that appear in multiple methods are prioritized.
    
    Args:
        df: Input dataframe
        target: Target column name
        k: Number of features to select
        voting_threshold: Minimum votes needed from methods to be selected
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (selected feature names, vote counts dictionary)
    """
    logging.info(f"\n{'='*60}")
    logging.info("ENSEMBLE FEATURE SELECTION")
    logging.info(f"{'='*60}")
    
    # Run all feature selection methods
    methods = {
        'correlation': select_top_k_by_corr(df, target, k),
        'mutual_info': select_by_mutual_info(df, target, k, random_state),
        'rf_importance': select_by_rf_importance(df, target, k, random_state),
        'rfe': select_by_rfe(df, target, k, random_state)
    }
    
    # Count votes for each feature
    all_features = []
    for method_name, features in methods.items():
        all_features.extend(features)
    
    vote_counts = Counter(all_features)
    
    # Select features with votes >= threshold
    selected_features = [feat for feat, votes in vote_counts.items() if votes >= voting_threshold]
    
    # If we don't have enough features, add top-voted ones
    if len(selected_features) < k:
        remaining = k - len(selected_features)
        candidates = [feat for feat, votes in vote_counts.most_common() if feat not in selected_features]
        selected_features.extend(candidates[:remaining])
    
    # If we have too many, keep only top k by vote count
    if len(selected_features) > k:
        selected_features = sorted(selected_features, key=lambda x: vote_counts[x], reverse=True)[:k]
    
    logging.info(f"\nVoting Results:")
    for feat in selected_features:
        logging.info(f"  - {feat}: {vote_counts[feat]} votes")
    
    logging.info(f"\nFinal Selection: {len(selected_features)} features")
    logging.info(f"{'='*60}\n")
    
    return selected_features, dict(vote_counts)


def select_features_advanced(
    df: pd.DataFrame,
    target: str,
    method: str = "ensemble",
    k: int = 8,
    voting_threshold: int = 2,
    random_state: int = 42
) -> Tuple[List[str], Optional[Dict[str, int]]]:
    """Advanced feature selection with multiple methods.
    
    Args:
        df: Input dataframe
        target: Target column name
        method: Selection method ("correlation", "mutual_info", "rf_importance", "rfe", "ensemble")
        k: Number of features to select
        voting_threshold: Minimum votes for ensemble method
        random_state: Random seed
        
    Returns:
        Tuple of (selected features, vote counts if ensemble else None)
    """
    if method == "correlation":
        return select_top_k_by_corr(df, target, k), None
    elif method == "mutual_info":
        return select_by_mutual_info(df, target, k, random_state), None
    elif method == "rf_importance":
        return select_by_rf_importance(df, target, k, random_state), None
    elif method == "rfe":
        return select_by_rfe(df, target, k, random_state), None
    elif method == "ensemble":
        return select_by_ensemble(df, target, k, voting_threshold, random_state)
    else:
        raise ValueError(f"Unknown feature selection method: {method}")


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
    
    # Determine which k to use (top_k_features takes precedence)
    k = cfg.top_k_features if cfg.top_k_features is not None else cfg.top_k_corr_features
    
    if k and k > 0:
        # Use advanced feature selection
        selected_features, vote_counts = select_features_advanced(
            df=df,
            target=cfg.target_col,
            method=cfg.feature_selection_method,
            k=k,
            voting_threshold=cfg.ensemble_voting_threshold,
            random_state=42
        )
        
        logging.info(f"Selected {len(selected_features)} features using '{cfg.feature_selection_method}' method")
        
        keep_cols = selected_features + [cfg.target_col]
        # Add id_cols only if they exist in dataframe
        if cfg.id_cols:
            keep_cols.extend([col for col in cfg.id_cols if col in df.columns])
        df = df[[col for col in keep_cols if col in df.columns]]
    else:
        # No feature selection - use all features
        selected_features = [c for c in df.columns if c not in (cfg.id_cols or []) + [cfg.target_col]]
        logging.info(f"No feature selection applied - using all {len(selected_features)} features")

    return df, selected_features
