"""
Production Experiment Runner
Orchestrates end-to-end training, evaluation, and artifact management.
"""
from __future__ import annotations
import os
import logging
import time
import json
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple
from datetime import datetime

# Import our refactored modules
from src.config_manager import ConfigManager, load_config
from src.preprocessing_v2 import preprocess_data, save_preprocessing_pipeline
from src.models_v2 import create_model
from src.evaluation import evaluate_model, optimize_threshold, save_evaluation_report

logger = logging.getLogger(__name__)


class ExperimentRunner:
    """
    Manages complete ML experiment lifecycle:
    - Data loading and splitting
    - Preprocessing pipeline fitting
    - Model training (classical and quantum)
    - Evaluation and metrics
    - Artifact saving
    - Results comparison
    """
    
    def __init__(self, config_path: str = "configs/config_master.yaml"):
        """
        Initialize experiment runner.
        
        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        self.config = load_config(config_path)
        self.config.set_global_seed()
        self.config.setup_logging()
        self.config.create_output_directories()
        
        # Experiment metadata
        self.experiment_id = self.config.experiment_id
        self.results = {}
        self.models = {}
        self.preprocessing_pipeline = None
        
        logger.info("="*80)
        logger.info(f"Experiment Runner Initialized: {self.experiment_id}")
        logger.info("="*80)
    
    def load_data(self) -> pd.DataFrame:
        """Load and merge transaction and identity data."""
        logger.info("Loading datasets...")
        
        data_config = self.config.get_section('data')
        
        # Load transaction data
        transaction_path = data_config['transaction_csv']
        nrows = data_config.get('nrows')
        
        if not os.path.exists(transaction_path):
            raise FileNotFoundError(f"Transaction file not found: {transaction_path}")
        
        df_transaction = pd.read_csv(transaction_path, nrows=nrows)
        logger.info(f"Loaded {len(df_transaction)} transactions")
        
        # Load identity data
        identity_path = data_config['identity_csv']
        if os.path.exists(identity_path):
            df_identity = pd.read_csv(identity_path, nrows=nrows)
            logger.info(f"Loaded {len(df_identity)} identity records")
            
            # Merge on TransactionID
            df_merged = pd.merge(
                df_transaction,
                df_identity,
                on='TransactionID',
                how='left'
            )
            logger.info(f"Merged dataset: {df_merged.shape}")
        else:
            logger.warning(f"Identity file not found: {identity_path}. Using transaction data only.")
            df_merged = df_transaction
        
        # Log data statistics
        target_col = data_config['target_col']
        fraud_rate = df_merged[target_col].mean()
        logger.info(f"Fraud rate: {fraud_rate:.2%} ({df_merged[target_col].sum()} frauds)")
        
        return df_merged
    
    def run_preprocessing(self, df: pd.DataFrame) -> Tuple:
        """
        Run preprocessing pipeline with leak prevention.
        
        Returns:
            pipeline, X_train, X_test, y_train, y_test
        """
        logger.info("Starting preprocessing pipeline...")
        
        # Get preprocessing config
        prep_config = {
            **self.config.get_section('preprocessing'),
            **self.config.get_section('data'),
            'random_seed': self.config.get('experiment.random_seed', 42)
        }
        
        # Run preprocessing (fits pipeline on train data only)
        pipeline, X_train, X_test, y_train, y_test = preprocess_data(
            df,
            config=prep_config,
            fit_pipeline=True
        )
        
        # Save pipeline
        pipeline_path = Path(self.config.get('paths.artifacts_dir')) / f"{self.experiment_id}_pipeline.pkl"
        save_preprocessing_pipeline(pipeline, str(pipeline_path))
        
        self.preprocessing_pipeline = pipeline
        
        logger.info(f"Preprocessing complete:")
        logger.info(f"  Train: {X_train.shape}, Fraud rate: {y_train.mean():.4f}")
        logger.info(f"  Test:  {X_test.shape}, Fraud rate: {y_test.mean():.4f}")
        
        return pipeline, X_train, X_test, y_train, y_test
    
    def train_classical_models(self, X_train, X_test, y_train, y_test):
        """Train all enabled classical models."""
        classical_config = self.config.get_section('classical_models')
        
        if not classical_config.get('enabled', True):
            logger.info("Classical models disabled. Skipping.")
            return
        
        logger.info("\n" + "="*80)
        logger.info("TRAINING CLASSICAL MODELS")
        logger.info("="*80)
        
        # Logistic Regression
        if classical_config.get('logistic_regression', {}).get('enabled', True):
            try:
                model = create_model(
                    'logistic_regression',
                    classical_config['logistic_regression']
                )
                model.fit(X_train.values, y_train.values)
                self.models['logistic_regression'] = model
                
                # Evaluate
                metrics = self._evaluate_and_save(
                    model, 'logistic_regression',
                    X_train, X_test, y_train, y_test
                )
                self.results['logistic_regression'] = metrics
            except Exception as e:
                logger.error(f"Logistic Regression failed: {e}", exc_info=True)
        
        # Random Forest
        if classical_config.get('random_forest', {}).get('enabled', False):
            try:
                model = create_model(
                    'random_forest',
                    classical_config['random_forest']
                )
                model.fit(X_train.values, y_train.values)
                self.models['random_forest'] = model
                
                metrics = self._evaluate_and_save(
                    model, 'random_forest',
                    X_train, X_test, y_train, y_test
                )
                self.results['random_forest'] = metrics
            except Exception as e:
                logger.error(f"Random Forest failed: {e}", exc_info=True)
        
        # XGBoost
        if classical_config.get('xgboost', {}).get('enabled', True):
            try:
                # Split training data for early stopping
                from sklearn.model_selection import train_test_split
                X_tr, X_val, y_tr, y_val = train_test_split(
                    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
                )
                
                model = create_model(
                    'xgboost',
                    classical_config['xgboost']
                )
                model.fit(X_tr.values, y_tr.values, X_val.values, y_val.values)
                self.models['xgboost'] = model
                
                metrics = self._evaluate_and_save(
                    model, 'xgboost',
                    X_train, X_test, y_train, y_test
                )
                self.results['xgboost'] = metrics
            except Exception as e:
                logger.error(f"XGBoost failed: {e}", exc_info=True)
    
    def train_quantum_models(self, X_train, X_test, y_train, y_test):
        """Train all enabled quantum models."""
        quantum_config = self.config.get_section('quantum_models')
        
        if not quantum_config.get('enabled', False):
            logger.info("Quantum models disabled. Skipping.")
            return
        
        logger.info("\n" + "="*80)
        logger.info("TRAINING QUANTUM MODELS")
        logger.info("="*80)
        
        n_features = X_train.shape[1]
        logger.info(f"Number of features (qubits): {n_features}")
        
        # Quantum VQC
        if quantum_config.get('vqc', {}).get('enabled', False):
            try:
                model = create_model(
                    'quantum_vqc',
                    quantum_config,
                    n_features=n_features
                )
                model.fit(X_train.values, y_train.values)
                self.models['quantum_vqc'] = model
                
                metrics = self._evaluate_and_save(
                    model, 'quantum_vqc',
                    X_train, X_test, y_train, y_test
                )
                self.results['quantum_vqc'] = metrics
            except Exception as e:
                logger.error(f"Quantum VQC failed: {e}", exc_info=True)
        
        # Quantum SVM
        if quantum_config.get('qsvm', {}).get('enabled', True):
            try:
                model = create_model(
                    'quantum_svm',
                    quantum_config,
                    n_features=n_features
                )
                model.fit(X_train.values, y_train.values)
                self.models['quantum_svm'] = model
                
                metrics = self._evaluate_and_save(
                    model, 'quantum_svm',
                    X_train, X_test, y_train, y_test
                )
                self.results['quantum_svm'] = metrics
            except Exception as e:
                logger.error(f"Quantum SVM failed: {e}", exc_info=True)
    
    def _evaluate_and_save(self, model, model_name, X_train, X_test, y_train, y_test) -> Dict:
        """Evaluate model and save artifacts."""
        from src.evaluation import evaluate_model, optimize_threshold
        
        # Predict
        y_pred = model.predict(X_test.values)
        y_proba = model.predict_proba(X_test.values)[:, 1]
        
        # Optimize threshold
        optimal_threshold, best_f1 = optimize_threshold(y_test.values, y_proba)
        logger.info(f"{model_name} - Optimal threshold: {optimal_threshold:.4f} (F1: {best_f1:.4f})")
        
        # Recompute predictions with optimal threshold
        y_pred_optimized = (y_proba >= optimal_threshold).astype(int)
        
        # Compute metrics
        metrics = evaluate_model(y_test.values, y_pred_optimized, y_proba)
        metrics['training_time'] = model.training_time
        metrics['optimal_threshold'] = optimal_threshold
        
        logger.info(f"{model_name} Results: AUC={metrics['roc_auc']:.4f}, "
                   f"F1={metrics['f1_score']:.4f}, Precision={metrics['precision']:.4f}")
        
        # Save model
        models_dir = Path(self.config.get('paths.models_dir'))
        model_path = models_dir / f"{self.experiment_id}_{model_name}.pkl"
        joblib.dump(model.model, model_path)
        logger.info(f"Model saved: {model_path}")
        
        # Save threshold
        threshold_path = models_dir / f"{self.experiment_id}_{model_name}_threshold.json"
        with open(threshold_path, 'w') as f:
            json.dump({'threshold': optimal_threshold, 'f1': best_f1}, f)
        
        return metrics
    
    def generate_comparison_report(self):
        """Generate comprehensive comparison report."""
        logger.info("\n" + "="*80)
        logger.info("RESULTS SUMMARY")
        logger.info("="*80)
        
        # Create results DataFrame
        df_results = pd.DataFrame(self.results).T
        df_results = df_results.sort_values('roc_auc', ascending=False)
        
        print(df_results.to_string())
        
        # Save to CSV
        results_path = Path(self.config.get('paths.results_dir')) / f"{self.experiment_id}_results.csv"
        df_results.to_csv(results_path)
        logger.info(f"Results saved: {results_path}")
        
        # Generate text report
        report_lines = []
        report_lines.append("="*80)
        report_lines.append(f"EXPERIMENT: {self.experiment_id}")
        report_lines.append(f"TIMESTAMP: {datetime.now().isoformat()}")
        report_lines.append("="*80)
        report_lines.append("")
        
        # Separate classical and quantum
        classical_models = {k: v for k, v in self.results.items() 
                          if not k.startswith('quantum')}
        quantum_models = {k: v for k, v in self.results.items() 
                        if k.startswith('quantum')}
        
        if classical_models:
            report_lines.append("CLASSICAL MODELS")
            report_lines.append("-"*80)
            for name, metrics in classical_models.items():
                report_lines.append(f"\n{name.upper()}:")
                report_lines.append(f"  ROC-AUC:    {metrics['roc_auc']:.4f}")
                report_lines.append(f"  F1 Score:   {metrics['f1_score']:.4f}")
                report_lines.append(f"  Precision:  {metrics['precision']:.4f}")
                report_lines.append(f"  Recall:     {metrics['recall']:.4f}")
                report_lines.append(f"  Training:   {metrics['training_time']:.2f}s")
        
        if quantum_models:
            report_lines.append("\n")
            report_lines.append("QUANTUM MODELS")
            report_lines.append("-"*80)
            for name, metrics in quantum_models.items():
                report_lines.append(f"\n{name.upper()}:")
                report_lines.append(f"  ROC-AUC:    {metrics['roc_auc']:.4f}")
                report_lines.append(f"  F1 Score:   {metrics['f1_score']:.4f}")
                report_lines.append(f"  Precision:  {metrics['precision']:.4f}")
                report_lines.append(f"  Recall:     {metrics['recall']:.4f}")
                report_lines.append(f"  Training:   {metrics['training_time']:.2f}s")
        
        # Quantum advantage analysis
        if classical_models and quantum_models:
            report_lines.append("\n")
            report_lines.append("QUANTUM ADVANTAGE ANALYSIS")
            report_lines.append("-"*80)
            
            best_classical_auc = max(m['roc_auc'] for m in classical_models.values())
            best_quantum_auc = max(m['roc_auc'] for m in quantum_models.values())
            
            improvement = ((best_quantum_auc - best_classical_auc) / best_classical_auc) * 100
            
            report_lines.append(f"Best Classical AUC: {best_classical_auc:.4f}")
            report_lines.append(f"Best Quantum AUC:   {best_quantum_auc:.4f}")
            report_lines.append(f"Improvement:        {improvement:+.2f}%")
        
        report_lines.append("\n" + "="*80)
        
        # Save report
        report_path = Path(self.config.get('paths.results_dir')) / f"{self.experiment_id}_report.txt"
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_lines))
        
        logger.info(f"Report saved: {report_path}")
        
        # Save config snapshot
        self.config.save_config_snapshot()
    
    def run(self):
        """Execute complete experiment."""
        try:
            start_time = time.time()
            
            # 1. Load data
            df = self.load_data()
            
            # 2. Preprocess
            pipeline, X_train, X_test, y_train, y_test = self.run_preprocessing(df)
            
            # 3. Train classical models
            self.train_classical_models(X_train, X_test, y_train, y_test)
            
            # 4. Train quantum models
            self.train_quantum_models(X_train, X_test, y_train, y_test)
            
            # 5. Generate reports
            self.generate_comparison_report()
            
            total_time = time.time() - start_time
            logger.info("\n" + "="*80)
            logger.info(f"EXPERIMENT COMPLETE: {self.experiment_id}")
            logger.info(f"Total time: {total_time:.2f}s ({total_time/60:.2f} minutes)")
            logger.info("="*80)
            
        except Exception as e:
            logger.error(f"Experiment failed: {e}", exc_info=True)
            raise


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Quantum Fraud Detection Experiment")
    parser.add_argument(
        '--config',
        type=str,
        default='configs/config_master.yaml',
        help='Path to configuration file'
    )
    
    args = parser.parse_args()
    
    # Run experiment
    runner = ExperimentRunner(config_path=args.config)
    runner.run()


if __name__ == "__main__":
    main()
