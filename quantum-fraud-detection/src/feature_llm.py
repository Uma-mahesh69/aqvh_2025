"""
LLM Feature Extractor for Financial Fraud Detection.
Uses Sentence-BERT to generate semantic embeddings from transaction metadata.
"""
import numpy as np
import pandas as pd
from typing import List, Optional, Union, Tuple
from sklearn.decomposition import PCA
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class LLMFeatureExtractor:
    """Extracts features using a pre-trained Encoder LLM (Sentence-BERT)."""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', n_components: int = 8, device: str = 'cpu'):
        """
        Initialize the LLM Feature Extractor.
        
        Args:
            model_name: Name of the sentence-transformers model.
            n_components: Number of PCA components to reduce embeddings to.
            device: 'cpu' or 'cuda'.
        """
        self.model_name = model_name
        self.n_components = n_components
        self.device = device
        self.model = None
        self.pca = None
        self._is_fitted = False
        
    def _load_model(self):
        """Lazy load the model to save memory if not used."""
        if self.model is None:
            try:
                from sentence_transformers import SentenceTransformer
                logging.info(f"Loading LLM model: {self.model_name}...")
                self.model = SentenceTransformer(self.model_name, device=self.device)
                logging.info("LLM model loaded successfully.")
            except ImportError:
                raise ImportError("sentence-transformers not installed. Run: pip install sentence-transformers")

    def _prepare_text(self, df: pd.DataFrame, text_cols: List[str]) -> List[str]:
        """
        Concatenate meaningful text columns into a single string per row.
        Handles missing values safely.
        """
        logging.info(f"Preparing text from columns: {text_cols}")
        
        # Ensure columns exist
        valid_cols = [c for c in text_cols if c in df.columns]
        if not valid_cols:
            logging.warning("No valid text columns found for LLM embedding. Returning empty strings.")
            return [""] * len(df)
            
        # Fill NaNs with empty string
        df_text = df[valid_cols].fillna("").astype(str)
        
        # Concatenate with feature names for context
        # Better formatting: "Transaction of {amt} with card {card}. Device info: {dev}."
        sentences = []
        for _, row in df_text.iterrows():
            # Build a narrative list
            parts = []
            
            # Prioritize standard columns if they exist (based on typical IEEE-CIS names)
            if 'TransactionAmt' in row:
                 parts.append(f"Transaction Amount: {row['TransactionAmt']}")
            if 'ProductCD' in row:
                 parts.append(f"Product Code: {row['ProductCD']}")
            if 'card4' in row and 'card6' in row:
                 parts.append(f"Card: {row['card4']} {row['card6']}")
            elif 'card4' in row:
                 parts.append(f"Card Network: {row['card4']}")
                 
            # Add all other columns genericly
            special_cols = ['TransactionAmt', 'ProductCD', 'card4', 'card6']
            context_parts = [f"{col} is {row[col]}" for col in valid_cols if row[col] and col not in special_cols]
            
            if context_parts:
                parts.extend(context_parts)
                
            # Create a reasoning-style sentence
            # "Transaction Amount: 50. Card: visa debit. P_emaildomain is gmail.com. This pattern represents..."
            text = ". ".join(parts) + "."
            sentences.append(text if parts else "Unknown Transaction Context.")
            
        return sentences

    def fit(self, df: pd.DataFrame, text_cols: List[str]):
        """
        Generate embeddings and fit PCA.
        """
        self._load_model()
        sentences = self._prepare_text(df, text_cols)
        
        logging.info(f"Generating embeddings for {len(sentences)} samples...")
        embeddings = self.model.encode(sentences, show_progress_bar=True)
        
        if len(embeddings) < self.n_components:
            logging.warning(f"Number of samples ({len(embeddings)}) < n_components ({self.n_components}). Reducing n_components.")
            self.n_components = min(len(embeddings), self.n_components)
            
        logging.info(f"Fitting PCA to reduce dimension from {embeddings.shape[1]} to {self.n_components}...")
        self.pca = PCA(n_components=self.n_components)
        self.pca.fit(embeddings)
        self._is_fitted = True
        return self

    def transform(self, df: pd.DataFrame, text_cols: List[str]) -> pd.DataFrame:
        """
        Generate embeddings and transform using fitted PCA.
        Returns a DataFrame with columns [llm_0, llm_1, ..., llm_n].
        """
        if not self._is_fitted:
            raise RuntimeError("LLMFeatureExtractor must be fitted before transform.")
            
        self._load_model()
        sentences = self._prepare_text(df, text_cols)
        
        logging.info(f"Generating embeddings for transformation ({len(sentences)} samples)...")
        embeddings = self.model.encode(sentences, show_progress_bar=True)
        
        reduced_embeddings = self.pca.transform(embeddings)
        
        cols = [f"llm_{i}" for i in range(self.n_components)]
        return pd.DataFrame(reduced_embeddings, columns=cols, index=df.index)

    def fit_transform(self, df: pd.DataFrame, text_cols: List[str]) -> pd.DataFrame:
        """Fit and transform in one step."""
        self.fit(df, text_cols)
        return self.transform(df, text_cols)


def add_llm_features(df: pd.DataFrame, config) -> Tuple[pd.DataFrame, 'LLMFeatureExtractor']:
    """
    Wrapper to add LLM features to dataframe.
    
    Args:
        df: Input dataframe
        config: PreprocessConfig object (or similar with llm_n_components)
        
    Returns:
        DataFrame with LLM features added
        Fitted LLMFeatureExtractor object
    """
    try:
        n_components = getattr(config, 'llm_n_components', 8)
        extractor = LLMFeatureExtractor(n_components=n_components)
        
        # Standard columns to use for context
        text_cols = ['TransactionAmt', 'ProductCD', 'card4', 'card6', 'P_emaildomain', 'R_emaildomain']
        
        # Fit transform
        df_llm = extractor.fit_transform(df, text_cols)
        
        return df_llm, extractor
        
    except Exception as e:
        logging.error(f"Error in add_llm_features: {e}")
        # Return empty dataframe with correct index if failure
        return pd.DataFrame(index=df.index), None
