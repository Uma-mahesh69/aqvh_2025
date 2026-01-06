from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from collections import Counter
import logging


@dataclass
class PreprocessConfig:
    missing_threshold: float = 50.0  # percent
    target_col: str = "isFraud"
    id_cols: Optional[List[str]] = None  # e.g., ["TransactionID"]
    top_k_corr_features: Optional[int] = 8  # Legacy (unused if method is 'pca')
    # Advanced feature selection options
    feature_selection_method: str = "pca"  # "pca", "ensemble", "correlation", "mutual_info", "rf_importance", "rfe"
    top_k_features: Optional[int] = 8  # Number of features to select or n_components for PCA
    ensemble_voting_threshold: int = 2  # Minimum votes needed from methods (for ensemble)
    
    # LLM Config
    use_llm_features: bool = False
    llm_n_components: int = 8


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates new, high-value features for the IEEE fraud dataset.
    Implements winning strategies from 1st place Kaggle solution.
    
    Key enhancements:
    - UID-based aggregations (most important)
    - Frequency encoding
    - Transaction amount splitting
    - Interaction features
    """
    logging.info("Starting ENHANCED feature engineering (NVIDIA insights)...")
    df_new = df.copy()

    # --- 1. Time-Based Features (from TransactionDT) ---
    start_date = pd.to_datetime('2017-12-01')
    df_new['Transaction_datetime'] = start_date + pd.to_timedelta(df_new['TransactionDT'], unit='s')
    
    # Extract time components
    df_new['hour_of_day'] = df_new['Transaction_datetime'].dt.hour
    df_new['day_of_week'] = df_new['Transaction_datetime'].dt.dayofweek
    df_new['day_of_month'] = df_new['Transaction_datetime'].dt.day
    df_new['is_weekend'] = (df_new['day_of_week'] >= 5).astype(int)
    
    # Calculate day number for UID creation
    df_new['day'] = df_new['TransactionDT'] / (24*60*60)

    # --- 2. Transaction Amount Features ---
    # NVIDIA Insight: Split dollars and cents for tree algorithms
    df_new['TransactionAmt_dollars'] = np.floor(df_new['TransactionAmt'])
    df_new['TransactionAmt_cents'] = (df_new['TransactionAmt'] - df_new['TransactionAmt_dollars']) * 100
    df_new['TransactionAmt_decimal'] = np.modf(df_new['TransactionAmt'])[0]
    df_new['TransactionAmt_log'] = np.log1p(df_new['TransactionAmt'])

    # --- 3. Enhanced User Identifier (UID) - CRITICAL FEATURE ---
    # NVIDIA Insight: UID from card1 + addr1 + D1 was the most important feature
    df_new['addr1'] = df_new['addr1'].fillna(-999)
    df_new['card1'] = df_new['card1'].fillna(-999)
    
    # Create robust UID (winning solution approach)
    if 'D1' in df_new.columns:
        df_new['D1'] = df_new['D1'].fillna(-999)
        df_new['UID'] = (df_new['card1'].astype(str) + '_' + 
                        df_new['addr1'].astype(str) + '_' + 
                        np.floor(df_new['day'] - df_new['D1']).astype(str))
    else:
        # Fallback if D1 not available
        df_new['UID'] = df_new['card1'].astype(str) + '_' + df_new['addr1'].astype(str)
    
    logging.info(f"Created UID with {df_new['UID'].nunique()} unique users")

    # --- 4. UID-Based Aggregations (MOST IMPORTANT) ---
    # NVIDIA Insight: These features detect unusual behavior for specific users
    
    # Transaction Amount aggregations
    df_new['TransactionAmt_UID_mean'] = df_new.groupby('UID')['TransactionAmt'].transform('mean')
    df_new['TransactionAmt_UID_std'] = df_new.groupby('UID')['TransactionAmt'].transform('std')
    df_new['amt_deviation_from_UID'] = (df_new['TransactionAmt'] - df_new['TransactionAmt_UID_mean']) / (df_new['TransactionAmt_UID_std'] + 1e-6)
    
    # D-column aggregations (timedelta features)
    d_cols = ['D1', 'D2', 'D4', 'D9', 'D10', 'D11', 'D15']
    for d_col in d_cols:
        if d_col in df_new.columns:
            df_new[f'{d_col}_UID_mean'] = df_new.groupby('UID')[d_col].transform('mean')
            df_new[f'{d_col}_UID_std'] = df_new.groupby('UID')[d_col].transform('std')
    
    # C-column aggregations (counting features)
    c_cols = ['C1', 'C2', 'C4', 'C5', 'C6', 'C13', 'C14']
    for c_col in c_cols:
        if c_col in df_new.columns:
            df_new[f'{c_col}_UID_mean'] = df_new.groupby('UID')[c_col].transform('mean')
    
    # UID transaction count (frequency)
    df_new['UID_transaction_count'] = df_new.groupby('UID')['TransactionID'].transform('count')
    
    logging.info(f"Created {len([c for c in df_new.columns if 'UID' in c])} UID-based features")

    # --- 5. Frequency Encoding ---
    # NVIDIA Insight: Count occurrences to detect rare values
    freq_encode_cols = ['card1', 'card2', 'card3', 'card4', 'card5', 'card6', 'addr1', 'addr2']
    for col in freq_encode_cols:
        if col in df_new.columns:
            freq_map = df_new[col].value_counts().to_dict()
            df_new[f'{col}_freq'] = df_new[col].map(freq_map)
    
    # Email domain frequency
    if 'P_emaildomain' in df_new.columns:
        df_new['P_emaildomain'] = df_new['P_emaildomain'].fillna('missing')
        freq_map = df_new['P_emaildomain'].value_counts().to_dict()
        df_new['P_emaildomain_freq'] = df_new['P_emaildomain'].map(freq_map)
        df_new[['p_email_provider', 'p_email_tld']] = df_new['P_emaildomain'].str.split('.', expand=True, n=1)
    
    if 'R_emaildomain' in df_new.columns:
        df_new['R_emaildomain'] = df_new['R_emaildomain'].fillna('missing')
        freq_map = df_new['R_emaildomain'].value_counts().to_dict()
        df_new['R_emaildomain_freq'] = df_new['R_emaildomain'].map(freq_map)
        df_new[['r_email_provider', 'r_email_tld']] = df_new['R_emaildomain'].str.split('.', expand=True, n=1)

    # --- 6. Interaction Features ---
    # NVIDIA Insight: Combine features that interact
    df_new['card1_addr1'] = df_new['card1'].astype(str) + '_' + df_new['addr1'].astype(str)
    
    if 'P_emaildomain' in df_new.columns:
        df_new['card1_addr1_P_emaildomain'] = df_new['card1_addr1'] + '_' + df_new['P_emaildomain'].astype(str)
        # Frequency encode interaction
        freq_map = df_new['card1_addr1_P_emaildomain'].value_counts().to_dict()
        df_new['card1_addr1_P_emaildomain_freq'] = df_new['card1_addr1_P_emaildomain'].map(freq_map)
    
    # Frequency encode card1_addr1
    freq_map = df_new['card1_addr1'].value_counts().to_dict()
    df_new['card1_addr1_freq'] = df_new['card1_addr1'].map(freq_map)

    # --- 7. Additional Aggregations by Interaction Features ---
    # Transaction amount by card1_addr1
    df_new['TransactionAmt_card1_addr1_mean'] = df_new.groupby('card1_addr1')['TransactionAmt'].transform('mean')
    df_new['TransactionAmt_card1_addr1_std'] = df_new.groupby('card1_addr1')['TransactionAmt'].transform('std')

    # Drop intermediate columns that shouldn't be used directly
    cols_to_drop = ['Transaction_datetime', 'UID', 'card1_addr1', 'card1_addr1_P_emaildomain']
    df_new = df_new.drop(columns=[c for c in cols_to_drop if c in df_new.columns], errors='ignore')
    
    logging.info(f"Feature engineering complete. Shape: {df_new.shape} ({df_new.shape[1] - df.shape[1]} new features)")
    return df_new


def drop_high_missing(df: pd.DataFrame, threshold_pct: float) -> pd.DataFrame:
    """Drop columns with missing values above threshold."""
    missing_pct = df.isnull().sum() / len(df) * 100
    cols_to_drop = missing_pct[missing_pct > threshold_pct].index.tolist()
    logging.info(f"Dropping {len(cols_to_drop)} columns with > {threshold_pct}% missing values.")
    return df.drop(columns=cols_to_drop)


def impute_simple(df: pd.DataFrame) -> pd.DataFrame:
    """Simple imputation: median for numeric, mode for categorical."""
    df = df.copy()
    for col in df.columns:
        if df[col].isnull().any():
            if pd.api.types.is_numeric_dtype(df[col]):
                imputer = SimpleImputer(strategy='median')
                # Convert to numpy array and back to handle 2D output from SimpleImputer
                df[col] = imputer.fit_transform(df[[col]].values).ravel()
            else:
                # For categorical, use the original approach
                mode_val = df[col].mode()
                df[col] = df[col].fillna(mode_val.iloc[0] if not mode_val.empty else "")
    return df


def label_encode_inplace(df: pd.DataFrame) -> pd.DataFrame:
    """Label encode all object-type columns."""
    df = df.copy()
    for col in df.select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
    return df


def scale_numeric(df: pd.DataFrame, exclude: Optional[List[str]] = None) -> Tuple[pd.DataFrame, StandardScaler]:
    """Scale numeric columns using StandardScaler (best for PCA)."""
    exclude = exclude or []
    df = df.copy()
    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude]
    scaler = StandardScaler()
    if num_cols:
        df[num_cols] = scaler.fit_transform(df[num_cols])
    return df, scaler


# --- Legacy Feature Selection (Kept for completeness) ---
# (These functions: select_top_k_by_corr, select_by_mutual_info, etc., are unchanged)

def select_top_k_by_corr(df: pd.DataFrame, target: str, k: int) -> List[str]:
    """Select top k features by absolute correlation with target."""
    if target not in df.columns:
        raise KeyError(f"Target '{target}' not in dataframe")
    corr = df.corr(numeric_only=True)[target].drop(labels=[target])
    top = corr.abs().sort_values(ascending=False).head(k).index.tolist()
    return top

def select_by_mutual_info(df: pd.DataFrame, target: str, k: int, random_state: int = 42) -> List[str]:
    """Select top k features by mutual information with target."""
    if target not in df.columns:
        raise KeyError(f"Target '{target}' not in dataframe")
    X = df.drop(columns=[target])
    y = df[target]
    mi_scores = mutual_info_classif(X, y, random_state=random_state)
    mi_scores = pd.Series(mi_scores, index=X.columns)
    top_features = mi_scores.sort_values(ascending=False).head(k).index.tolist()
    logging.info(f"Mutual Info - Top {k} features: {top_features}")
    return top_features

def select_by_rf_importance(df: pd.DataFrame, target: str, k: int, random_state: int = 42) -> List[str]:
    """Select top k features by Random Forest feature importance."""
    if target not in df.columns:
        raise KeyError(f"Target '{target}' not in dataframe")
    X = df.drop(columns=[target])
    y = df[target]
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=random_state, n_jobs=-1)
    rf.fit(X, y)
    importances = pd.Series(rf.feature_importances_, index=X.columns)
    top_features = importances.sort_values(ascending=False).head(k).index.tolist()
    logging.info(f"RF Importance - Top {k} features: {top_features}")
    return top_features

def select_by_rfe(df: pd.DataFrame, target: str, k: int, random_state: int = 42) -> List[str]:
    """Select top k features using Recursive Feature Elimination (RFE)."""
    if target not in df.columns:
        raise KeyError(f"Target '{target}' not in dataframe")
    X = df.drop(columns=[target])
    y = df[target]
    estimator = LogisticRegression(max_iter=1000, random_state=random_state)
    rfe = RFE(estimator=estimator, n_features_to_select=k, step=1)
    rfe.fit(X, y)
    top_features = X.columns[rfe.support_].tolist()
    logging.info(f"RFE - Top {k} features: {top_features}")
    return top_features

def select_by_ensemble(df: pd.DataFrame, target: str, k: int, voting_threshold: int = 2, random_state: int = 42) -> Tuple[List[str], Dict[str, int]]:
    """Select features using ensemble voting from multiple methods."""
    logging.info(f"\n{'='*60}\nENSEMBLE FEATURE SELECTION\n{'='*60}")
    methods = {
        'correlation': select_top_k_by_corr(df, target, k),
        'mutual_info': select_by_mutual_info(df, target, k, random_state),
        'rf_importance': select_by_rf_importance(df, target, k, random_state),
        'rfe': select_by_rfe(df, target, k, random_state)
    }
    all_features = [feat for features in methods.values() for feat in features]
    vote_counts = Counter(all_features)
    selected_features = [feat for feat, votes in vote_counts.items() if votes >= voting_threshold]
    
    if len(selected_features) < k:
        remaining = k - len(selected_features)
        candidates = [feat for feat, _ in vote_counts.most_common() if feat not in selected_features]
        selected_features.extend(candidates[:remaining])
    
    if len(selected_features) > k:
        selected_features = sorted(selected_features, key=lambda x: vote_counts[x], reverse=True)[:k]
    
    logging.info(f"\nVoting Results (Top {len(selected_features)}):")
    for feat in selected_features:
        logging.info(f"  - {feat}: {vote_counts[feat]} votes")
    logging.info(f"{'='*60}\n")
    return selected_features, dict(vote_counts)

def select_by_l1(df: pd.DataFrame, target: str, k: int, random_state: int = 42) -> List[str]:
    """Select top k features using L1 regularization (Lasso)."""
    if target not in df.columns:
        raise KeyError(f"Target '{target}' not in dataframe")
    X = df.drop(columns=[target])
    y = df[target]
    
    # Use Logistic Regression with L1 penalty
    # C=0.1 means strong regularization -> sparse solution
    model = LogisticRegression(penalty='l1', C=0.1, solver='liblinear', random_state=random_state)
    model.fit(X, y)
    
    coefs = pd.Series(np.abs(model.coef_[0]), index=X.columns)
    top_features = coefs.sort_values(ascending=False).head(k).index.tolist()
    logging.info(f"L1 (Lasso) - Top {k} features: {top_features}")
    return top_features

def select_features_advanced(
    df: pd.DataFrame, target: str, method: str = "ensemble", k: int = 8,
    voting_threshold: int = 2, random_state: int = 42
) -> Tuple[List[str], Optional[Dict[str, int]]]:
    """Dispatcher for feature selection methods."""
    if method == "pca":
        # PCA is handled separately in preprocess_pipeline
        return [], None
    elif method == "correlation":
        return select_top_k_by_corr(df, target, k), None
    elif method == "mutual_info":
        return select_by_mutual_info(df, target, k, random_state), None
    elif method == "rf_importance":
        return select_by_rf_importance(df, target, k, random_state), None
    elif method == "rfe":
        return select_by_rfe(df, target, k, random_state), None
    elif method == "l1":
        return select_by_l1(df, target, k, random_state), None
    elif method == "ensemble":
        return select_by_ensemble(df, target, k, voting_threshold, random_state)
    else:
        raise ValueError(f"Unknown feature selection method: {method}")

# --- END Legacy Functions ---


def split_data(
    df: pd.DataFrame,
    target: str,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split data into train and test sets (random split - legacy)."""
    X = df.drop(columns=[target])
    y = df[target]
    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if stratify else None,
    )


def split_data_time_based(
    df: pd.DataFrame,
    target: str,
    time_col: str = 'TransactionDT',
    test_size: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data chronologically for time-series fraud detection.
    
    NVIDIA Insight: Time-based validation prevents data leakage and reflects
    real-world deployment where models predict future transactions.
    
    Args:
        df: Input dataframe
        target: Target column name
        time_col: Time column to sort by (default: TransactionDT)
        test_size: Fraction of data for test set
    
    Returns:
        X_train, X_test, y_train, y_test
    """
    if time_col not in df.columns:
        logging.warning(f"Time column '{time_col}' not found. Falling back to random split.")
        return split_data(df, target, test_size=test_size, stratify=False)
    
    # Sort by time
    df_sorted = df.sort_values(time_col).reset_index(drop=True)
    
    # Split chronologically
    split_idx = int(len(df_sorted) * (1 - test_size))
    train_df = df_sorted.iloc[:split_idx]
    test_df = df_sorted.iloc[split_idx:]
    
    # Drop time column from features
    X_train = train_df.drop(columns=[target, time_col])
    X_test = test_df.drop(columns=[target, time_col])
    y_train = train_df[target]
    y_test = test_df[target]
    
    logging.info(f"Time-based split: Train={len(train_df)} ({y_train.sum()} fraud), Test={len(test_df)} ({y_test.sum()} fraud)")
    logging.info(f"Train fraud rate: {y_train.mean():.4f}, Test fraud rate: {y_test.mean():.4f}")
    
    return X_train, X_test, y_train, y_test


def preprocess_pipeline(
    df_merged: pd.DataFrame,
    cfg: PreprocessConfig,
) -> Tuple[pd.DataFrame, List[str], Dict]:
    """
    Complete preprocessing pipeline:
    1. Feature Engineering
    2. Cleaning (Drop high missing, Impute)
    3. Encoding (Label Encode)
    4. Scaling (StandardScaler)
    5. Dimensionality Reduction (PCA) or legacy Feature Selection
    """
    # --- 1. Feature Engineering ---
    df = engineer_features(df_merged)

    # --- 1.5 LLM Feature Extraction (New) ---
    if cfg.use_llm_features:
        logging.info("Starting LLM Feature Extraction...")
        try:
            from .feature_llm import LLMFeatureExtractor
            # Define text columns to use
            text_cols = ['DeviceInfo', 'P_emaildomain', 'R_emaildomain', 'id_30', 'id_31', 'DeviceType']
            
            # Initialize Extractor
            llm_extractor = LLMFeatureExtractor(n_components=cfg.llm_n_components)
            
            # Generate Embeddings (returns DataFrame with 'llm_0', 'llm_1'...)
            df_llm = llm_extractor.fit_transform(df, text_cols)
            
            # Concatenate features
            original_len = len(df.columns)
            df = pd.concat([df, df_llm], axis=1)
            logging.info(f"Added {len(df_llm.columns)} LLM features. Total cols: {original_len} -> {len(df.columns)}")
            
        except Exception as e:
            logging.warning(f"LLM Feature Extraction failed: {e}. Proceeding without LLM features.")

    # --- 2. Cleaning ---
    df = drop_high_missing(df, cfg.missing_threshold)
    df = impute_simple(df) # Impute *after* engineering to catch NaNs created by splits/etc.

    # --- 3. Encoding ---
    df = label_encode_inplace(df)

    # --- 4. Scaling ---
    # Preserve TransactionDT for time-based validation (don't scale it)
    exclude_cols = [cfg.target_col] + (cfg.id_cols or [])
    if 'TransactionDT' in df.columns and 'TransactionDT' not in exclude_cols:
        exclude_cols.append('TransactionDT')
    
    df, scaler = scale_numeric(df, exclude=exclude_cols)
    
    # Define features to be used for PCA or selection (exclude TransactionDT from features)
    all_features = [c for c in df.columns if c not in exclude_cols]
    
    # Define features to be used for PCA or selection (exclude TransactionDT from features)
    all_features = [c for c in df.columns if c not in exclude_cols]
    
    # Collect artifacts for inference (initialize)
    artifacts = {
        'scaler': scaler,
        'exclude_cols': exclude_cols
    }
    
    # --- 5. Dimensionality Reduction (PCA) or Feature Selection ---
    k = cfg.top_k_features if cfg.top_k_features is not None else len(all_features)
    
    if cfg.feature_selection_method == "pca":
        logging.info(f"Applying PCA to reduce {len(all_features)} features to {k} components.")
        pca = PCA(n_components=k, random_state=42)
        
        # Create a dataframe for the PCA components
        X_pca = pca.fit_transform(df[all_features])
        pca_cols = [f"PCA_{i}" for i in range(k)]
        df_pca = pd.DataFrame(X_pca, columns=pca_cols, index=df.index)
        
        # Combine PCA components with target/id columns
        df_final = pd.concat([df[exclude_cols], df_pca], axis=1)
        selected_features = pca_cols
        
        artifacts['pca'] = pca
        artifacts['pca_cols'] = pca_cols
        artifacts['pca_input_features'] = all_features
        
    elif k > 0:
        # --- Legacy Feature Selection ---
        logging.warning(f"Using legacy feature selection method: '{cfg.feature_selection_method}'. 'pca' is recommended.")
        selected_features, _ = select_features_advanced(
            df=df,
            target=cfg.target_col,
            method=cfg.feature_selection_method,
            k=k,
            voting_threshold=cfg.ensemble_voting_threshold,
            random_state=42
        )
        
        logging.info(f"Selected {len(selected_features)} features using '{cfg.feature_selection_method}' method")
        keep_cols = selected_features + exclude_cols
        df_final = df[[col for col in keep_cols if col in df.columns]]
        
    else:
        # No feature selection - use all features
        logging.info(f"No feature selection/reduction applied - using all {len(all_features)} features")
        selected_features = all_features
        df_final = df

    # Ensure target column is int
    df_final[cfg.target_col] = df_final[cfg.target_col].astype(int)
    
    if 'llm_extractor' in locals():
        artifacts['llm_extractor'] = llm_extractor
        
    artifacts['selected_features'] = selected_features
    
    return df_final, selected_features, artifacts
