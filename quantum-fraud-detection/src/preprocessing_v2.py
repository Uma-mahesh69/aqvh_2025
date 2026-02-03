"""
Production-Grade Preprocessing Pipeline
Eliminates data leakage through proper sklearn Pipeline integration.
All transformers fit ONLY on training data.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import logging
from typing import Tuple, List, Dict, Any, Optional
from dataclasses import dataclass
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import joblib

logger = logging.getLogger(__name__)


# ==============================================================================
# CUSTOM TRANSFORMERS (Sklearn-Compatible)
# ==============================================================================

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Creates engineered features from raw data.
    CRITICAL: Only uses per-row transformations or training set statistics.
    """
    
    def __init__(
        self,
        create_time_features: bool = True,
        create_amount_features: bool = True,
        create_frequency_features: bool = True,
        target_col: str = "isFraud"
    ):
        self.create_time_features = create_time_features
        self.create_amount_features = create_amount_features
        self.create_frequency_features = create_frequency_features
        self.target_col = target_col
        
        # Fitted statistics (MUST be fit on training data only)
        self.freq_encodings_ = {}
        self.is_fitted_ = False
    
    def fit(self, X: pd.DataFrame, y=None):
        """
        Fit frequency encodings and other statistics on training data.
        
        Args:
            X: Training features (DataFrame)
            y: Training labels (ignored, kept for sklearn compatibility)
        """
        logger.info("Fitting FeatureEngineer transformer...")
        X = X.copy()
        
        # Calculate frequency encodings on training set
        if self.create_frequency_features:
            freq_cols = ['card1', 'card2', 'card3', 'card4', 'card5', 'card6', 
                        'addr1', 'addr2', 'P_emaildomain', 'R_emaildomain']
            
            for col in freq_cols:
                if col in X.columns:
                    self.freq_encodings_[col] = X[col].value_counts().to_dict()
        
        self.is_fitted_ = True
        logger.info(f"FeatureEngineer fitted. Freq encodings for {len(self.freq_encodings_)} columns.")
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform features using fitted statistics.
        
        Args:
            X: Features to transform
        
        Returns:
            Transformed DataFrame
        """
        if not self.is_fitted_:
            raise RuntimeError("FeatureEngineer must be fitted before transform")
        
        X = X.copy()
        
        # --- 1. Time Features ---
        if self.create_time_features and 'TransactionDT' in X.columns:
            start_date = pd.to_datetime('2017-12-01')
            X['Transaction_datetime'] = start_date + pd.to_timedelta(X['TransactionDT'], unit='s')
            X['hour_of_day'] = X['Transaction_datetime'].dt.hour
            X['day_of_week'] = X['Transaction_datetime'].dt.dayofweek
            X['day_of_month'] = X['Transaction_datetime'].dt.day
            X['is_weekend'] = (X['day_of_week'] >= 5).astype(int)
            X['day'] = X['TransactionDT'] / (24 * 60 * 60)
            X.drop('Transaction_datetime', axis=1, inplace=True)
        
        # --- 2. Amount Features ---
        if self.create_amount_features and 'TransactionAmt' in X.columns:
            X['TransactionAmt_dollars'] = np.floor(X['TransactionAmt'])
            X['TransactionAmt_cents'] = (X['TransactionAmt'] - X['TransactionAmt_dollars']) * 100
            X['TransactionAmt_decimal'] = np.modf(X['TransactionAmt'])[0]
            X['TransactionAmt_log'] = np.log1p(X['TransactionAmt'])
        
        # --- 3. Frequency Features (using fitted encodings) ---
        if self.create_frequency_features:
            for col, freq_map in self.freq_encodings_.items():
                if col in X.columns:
                    # Use .get() to handle unseen categories gracefully
                    X[f'{col}_freq'] = X[col].map(lambda x: freq_map.get(x, 0))
        
        # --- 4. Email Domain Parsing ---
        for email_col in ['P_emaildomain', 'R_emaildomain']:
            if email_col in X.columns:
                X[email_col] = X[email_col].fillna('missing')
                prefix = email_col.split('_')[0].lower()
                
                # Split domain
                split_result = X[email_col].str.split('.', expand=True, n=1)
                if split_result.shape[1] >= 2:
                    X[f'{prefix}_email_provider'] = split_result[0]
                    X[f'{prefix}_email_tld'] = split_result[1]
                else:
                    X[f'{prefix}_email_provider'] = split_result[0]
                    X[f'{prefix}_email_tld'] = 'missing'
        
        return X
    
    def get_feature_names_out(self, input_features=None):
        """Return feature names for output."""
        # This would need to be implemented properly for full sklearn compatibility
        return input_features


class ColumnDropper(BaseEstimator, TransformerMixin):
    """Drops specified columns and high-missing columns."""
    
    def __init__(self, missing_threshold: float = 50.0, columns_to_drop: List[str] = None):
        self.missing_threshold = missing_threshold
        self.columns_to_drop = columns_to_drop or []
        self.high_missing_cols_ = []
    
    def fit(self, X: pd.DataFrame, y=None):
        """Identify columns to drop based on training data."""
        # Find columns with high missing rate
        missing_pct = X.isnull().sum() / len(X) * 100
        self.high_missing_cols_ = missing_pct[missing_pct > self.missing_threshold].index.tolist()
        logger.info(f"Identified {len(self.high_missing_cols_)} columns with >{self.missing_threshold}% missing")
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Drop identified columns."""
        X = X.copy()
        cols_to_drop = list(set(self.columns_to_drop + self.high_missing_cols_))
        cols_to_drop = [c for c in cols_to_drop if c in X.columns]
        return X.drop(columns=cols_to_drop, errors='ignore')


class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """
    Encodes categorical columns with LabelEncoder.
    Handles unseen categories in test set gracefully.
    """
    
    def __init__(self, handle_unknown: str = 'use_encoded_value', unknown_value: int = -1):
        self.handle_unknown = handle_unknown
        self.unknown_value = unknown_value
        self.encoders_ = {}
        self.categorical_columns_ = []
    
    def fit(self, X: pd.DataFrame, y=None):
        """Fit label encoders on training data."""
        self.categorical_columns_ = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        for col in self.categorical_columns_:
            le = LabelEncoder()
            # Fit on non-null values
            non_null_vals = X[col].dropna().astype(str)
            le.fit(non_null_vals)
            self.encoders_[col] = le
        
        logger.info(f"Fitted encoders for {len(self.encoders_)} categorical columns")
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform categorical columns using fitted encoders."""
        X = X.copy()
        
        for col in self.categorical_columns_:
            if col not in X.columns:
                continue
            
            le = self.encoders_[col]
            
            # Handle unseen categories
            def encode_with_unknown(x):
                if pd.isna(x):
                    return self.unknown_value
                x_str = str(x)
                if x_str in le.classes_:
                    return le.transform([x_str])[0]
                else:
                    return self.unknown_value
            
            X[col] = X[col].apply(encode_with_unknown)
        
        return X


class DataFrameImputer(BaseEstimator, TransformerMixin):
    """Imputes missing values in DataFrame, handling both numeric and categorical."""
    
    def __init__(self, strategy: str = 'median'):
        self.strategy = strategy
        self.imputers_ = {}
        self.columns_ = []
    
    def fit(self, X: pd.DataFrame, y=None):
        """Fit imputers for each column."""
        self.columns_ = X.columns.tolist()
        
        for col in X.columns:
            if X[col].isnull().any():
                if pd.api.types.is_numeric_dtype(X[col]):
                    imputer = SimpleImputer(strategy=self.strategy)
                    imputer.fit(X[[col]])
                    self.imputers_[col] = imputer
                else:
                    # For categorical, use mode
                    mode_val = X[col].mode()
                    self.imputers_[col] = mode_val[0] if len(mode_val) > 0 else ""
        
        logger.info(f"Fitted imputers for {len(self.imputers_)} columns")
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Impute missing values using fitted statistics."""
        X = X.copy()
        
        for col, imputer in self.imputers_.items():
            if col not in X.columns:
                continue
            
            if X[col].isnull().any():
                if isinstance(imputer, SimpleImputer):
                    X[col] = imputer.transform(X[[col]]).ravel()
                else:
                    X[col] = X[col].fillna(imputer)
        
        return X


class DataFrameScaler(BaseEstimator, TransformerMixin):
    """Scales numeric columns in DataFrame."""
    
    def __init__(self, scaler_type: str = 'standard', exclude_cols: List[str] = None):
        self.scaler_type = scaler_type
        self.exclude_cols = exclude_cols or []
        self.scaler_ = None
        self.numeric_cols_ = []
    
    def fit(self, X: pd.DataFrame, y=None):
        """Fit scaler on numeric columns."""
        # Identify numeric columns to scale
        self.numeric_cols_ = [
            col for col in X.select_dtypes(include=[np.number]).columns
            if col not in self.exclude_cols
        ]
        
        if not self.numeric_cols_:
            logger.warning("No numeric columns to scale")
            return self
        
        # Create scaler
        if self.scaler_type == 'standard':
            self.scaler_ = StandardScaler()
        elif self.scaler_type == 'minmax':
            self.scaler_ = MinMaxScaler()
        elif self.scaler_type == 'robust':
            self.scaler_ = RobustScaler()
        else:
            raise ValueError(f"Unknown scaler type: {self.scaler_type}")
        
        # Fit on numeric columns
        self.scaler_.fit(X[self.numeric_cols_])
        logger.info(f"Fitted {self.scaler_type} scaler on {len(self.numeric_cols_)} columns")
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Scale numeric columns."""
        if not self.numeric_cols_ or self.scaler_ is None:
            return X
        
        X = X.copy()
        X[self.numeric_cols_] = self.scaler_.transform(X[self.numeric_cols_])
        return X


class DimensionalityReducer(BaseEstimator, TransformerMixin):
    """Applies PCA or feature selection for dimensionality reduction."""
    
    def __init__(
        self,
        method: str = 'pca',
        n_components: int = 8,
        variance_threshold: float = 0.01,
        exclude_cols: List[str] = None
    ):
        self.method = method
        self.n_components = n_components
        self.variance_threshold = variance_threshold
        self.exclude_cols = exclude_cols or []
        self.reducer_ = None
        self.feature_cols_ = []
        self.output_cols_ = []
    
    def fit(self, X: pd.DataFrame, y=None):
        """Fit dimensionality reducer."""
        # Get columns to transform (exclude target, IDs, etc.)
        self.feature_cols_ = [col for col in X.columns if col not in self.exclude_cols]
        
        if not self.feature_cols_:
            logger.warning("No features to reduce")
            return self
        
        if self.method == 'pca':
            n_comp = min(self.n_components, len(self.feature_cols_), len(X))
            self.reducer_ = PCA(n_components=n_comp, random_state=42)
            self.reducer_.fit(X[self.feature_cols_])
            self.output_cols_ = [f'PCA_{i}' for i in range(n_comp)]
            
            explained_var = self.reducer_.explained_variance_ratio_.sum()
            logger.info(f"PCA: {n_comp} components explain {explained_var:.2%} variance")
        
        elif self.method == 'none':
            self.output_cols_ = self.feature_cols_
            logger.info("No dimensionality reduction applied")
        
        else:
            raise ValueError(f"Unknown reduction method: {self.method}")
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform features."""
        if not self.feature_cols_:
            return X
        
        if self.method == 'pca' and self.reducer_ is not None:
            X_reduced = self.reducer_.transform(X[self.feature_cols_])
            df_reduced = pd.DataFrame(X_reduced, columns=self.output_cols_, index=X.index)
            
            # Combine with excluded columns
            excluded_data = X[self.exclude_cols] if self.exclude_cols else pd.DataFrame(index=X.index)
            return pd.concat([excluded_data, df_reduced], axis=1)
        
        elif self.method == 'none':
            return X
        
        return X


# ==============================================================================
# PIPELINE BUILDER
# ==============================================================================

def build_preprocessing_pipeline(config: Dict[str, Any]) -> Pipeline:
    """
    Build sklearn Pipeline for preprocessing.
    All transformers fit ONLY on training data.
    
    Args:
        config: Preprocessing configuration from config manager
    
    Returns:
        Sklearn Pipeline
    """
    target_col = config.get('target_col', 'isFraud')
    id_cols = config.get('id_cols', ['TransactionID'])
    
    # Get sub-configs
    feat_eng_config = config.get('feature_engineering', {})
    feat_sel_config = config.get('feature_selection', {})
    
    # Columns to exclude from transformations
    # Columns to exclude from transformations (IDs, Time)
    # Note: target_col is already separated from X, so we don't need to exclude it here.
    exclude_cols = list(id_cols)
    if config.get('use_time_based_split', False):
        exclude_cols.append(config.get('time_col', 'TransactionDT'))
    
    steps = []
    
    # Step 1: Feature Engineering
    if feat_eng_config.get('enabled', True):
        steps.append(('feature_engineering', FeatureEngineer(
            create_time_features=feat_eng_config.get('create_time_features', True),
            create_amount_features=feat_eng_config.get('create_amount_features', True),
            create_frequency_features=feat_eng_config.get('create_frequency_features', True),
            target_col=target_col
        )))
    
    # Step 2: Drop high-missing columns
    steps.append(('drop_columns', ColumnDropper(
        missing_threshold=config.get('missing_threshold', 50.0),
        columns_to_drop=[]
    )))
    
    # Step 3: Imputation
    steps.append(('imputation', DataFrameImputer(
        strategy=config.get('imputation_strategy', 'median')
    )))
    
    # Step 4: Categorical Encoding
    steps.append(('encoding', CategoricalEncoder(
        handle_unknown='use_encoded_value',
        unknown_value=-1
    )))
    
    # Step 5: Scaling
    steps.append(('scaling', DataFrameScaler(
        scaler_type=config.get('scaler', 'standard'),
        exclude_cols=exclude_cols
    )))
    
    # Step 6: Dimensionality Reduction
    steps.append(('dimensionality_reduction', DimensionalityReducer(
        method=feat_sel_config.get('method', 'pca'),
        n_components=feat_sel_config.get('n_components', 8),
        variance_threshold=feat_sel_config.get('variance_threshold', 0.01),
        exclude_cols=exclude_cols
    )))
    
    pipeline = Pipeline(steps)
    logger.info(f"Built preprocessing pipeline with {len(steps)} steps")
    
    return pipeline


def split_data_safely(
    df: pd.DataFrame,
    target_col: str,
    use_time_based: bool = True,
    time_col: str = 'TransactionDT',
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data into train/test sets with proper temporal ordering.
    
    Args:
        df: Full dataset
        target_col: Target column name
        use_time_based: Use chronological split (prevents leakage)
        time_col: Time column for sorting
        test_size: Proportion of test set
        random_state: Random seed
    
    Returns:
        X_train, X_test, y_train, y_test
    """
    if use_time_based and time_col in df.columns:
        # Chronological split
        logger.info("Performing time-based split to prevent temporal leakage...")
        df_sorted = df.sort_values(time_col).reset_index(drop=True)
        
        split_idx = int(len(df_sorted) * (1 - test_size))
        train_df = df_sorted.iloc[:split_idx]
        test_df = df_sorted.iloc[split_idx:]
        
        X_train = train_df.drop(columns=[target_col])
        X_test = test_df.drop(columns=[target_col])
        y_train = train_df[target_col]
        y_test = test_df[target_col]
        
        logger.info(f"Time-based split: Train fraud rate={y_train.mean():.4f}, "
                   f"Test fraud rate={y_test.mean():.4f}")
    else:
        # Random stratified split
        logger.info("Performing stratified random split...")
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )
    
    logger.info(f"Train set: {len(X_train)} samples ({y_train.sum()} fraud)")
    logger.info(f"Test set: {len(X_test)} samples ({y_test.sum()} fraud)")
    
    return X_train, X_test, y_train, y_test


# ==============================================================================
# MAIN PREPROCESSING FUNCTION
# ==============================================================================

def preprocess_data(
    df: pd.DataFrame,
    config: Dict[str, Any],
    fit_pipeline: bool = True,
    pipeline: Optional[Pipeline] = None
) -> Tuple[Pipeline, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Complete preprocessing workflow with leak prevention.
    
    Args:
        df: Raw merged dataset
        config: Preprocessing configuration
        fit_pipeline: Whether to fit new pipeline (True for training, False for inference)
        pipeline: Pre-fitted pipeline (for inference)
    
    Returns:
        pipeline, X_train, X_test, y_train, y_test
    """
    target_col = config.get('target_col', 'isFraud')
    
    # Step 1: Split BEFORE any transformations
    X_train_raw, X_test_raw, y_train, y_test = split_data_safely(
        df,
        target_col=target_col,
        use_time_based=config.get('use_time_based_split', True),
        time_col=config.get('time_col', 'TransactionDT'),
        test_size=config.get('test_size', 0.2),
        random_state=config.get('random_seed', 42)
    )
    
    # Step 2: Build or use existing pipeline
    if fit_pipeline:
        logger.info("Building and fitting preprocessing pipeline...")
        pipeline = build_preprocessing_pipeline(config)
        
        # Fit on training data ONLY
        X_train_transformed = pipeline.fit_transform(X_train_raw)
    else:
        if pipeline is None:
            raise ValueError("Must provide fitted pipeline when fit_pipeline=False")
        logger.info("Using pre-fitted pipeline...")
        X_train_transformed = X_train_raw
    
    # Step 3: Transform test data using fitted pipeline
    X_test_transformed = pipeline.transform(X_test_raw)
    
    # Ensure DataFrames
    if not isinstance(X_train_transformed, pd.DataFrame):
        X_train_transformed = pd.DataFrame(X_train_transformed)
    if not isinstance(X_test_transformed, pd.DataFrame):
        X_test_transformed = pd.DataFrame(X_test_transformed)
    
    logger.info(f"Preprocessing complete. Train shape: {X_train_transformed.shape}, "
               f"Test shape: {X_test_transformed.shape}")
    
    return pipeline, X_train_transformed, X_test_transformed, y_train, y_test


def save_preprocessing_pipeline(pipeline: Pipeline, output_path: str):
    """Save fitted preprocessing pipeline."""
    joblib.dump(pipeline, output_path)
    logger.info(f"Preprocessing pipeline saved to {output_path}")


def load_preprocessing_pipeline(input_path: str) -> Pipeline:
    """Load fitted preprocessing pipeline."""
    pipeline = joblib.load(input_path)
    logger.info(f"Preprocessing pipeline loaded from {input_path}")
    return pipeline
