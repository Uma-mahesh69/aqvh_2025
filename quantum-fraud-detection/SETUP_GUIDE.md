# Quantum Fraud Detection - Complete Setup Guide

This guide will walk you through setting up and running the complete quantum fraud detection pipeline.

## 📋 Prerequisites

- Python 3.9 or higher
- pip package manager
- 8GB+ RAM recommended
- IBM Quantum account (for real hardware execution)

## 🔧 Step-by-Step Setup

### 1. Environment Setup

```bash
# Navigate to project directory
cd quantum-fraud-detection

# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip
```

### 2. Install Dependencies

```bash
# Install all required packages
pip install -r requirements.txt
```

**Expected packages:**
- pandas, numpy, scikit-learn (classical ML)
- xgboost (gradient boosting)
- imbalanced-learn (handling imbalanced data)
- matplotlib, seaborn (visualization)
- qiskit, qiskit-aer (quantum simulation)
- qiskit-machine-learning (quantum ML algorithms)
- qiskit-ibm-runtime (IBM Quantum hardware access)

### 3. Data Preparation

**Option A: Download from Kaggle**

1. Go to [IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection/data)
2. Download `train_transaction.csv` and `train_identity.csv`
3. Place them in the `data/` directory:
   ```
   quantum-fraud-detection/
   └── data/
       ├── train_transaction.csv
       └── train_identity.csv
   ```

**Option B: Use Sample Data**

For testing, you can create a smaller sample:
```python
import pandas as pd

# Load full dataset
df_txn = pd.read_csv('path/to/train_transaction.csv')
df_id = pd.read_csv('path/to/train_identity.csv')

# Sample 10,000 transactions
df_txn_sample = df_txn.sample(n=10000, random_state=42)
df_id_sample = df_id[df_id['TransactionID'].isin(df_txn_sample['TransactionID'])]

# Save samples
df_txn_sample.to_csv('data/train_transaction.csv', index=False)
df_id_sample.to_csv('data/train_identity.csv', index=False)
```

### 4. IBM Quantum Setup (Optional - for Real Hardware)

**Step 4.1: Create IBM Quantum Account**

1. Visit [IBM Quantum](https://quantum.ibm.com/)
2. Click "Sign In" or "Create Account"
3. Complete registration

**Step 4.2: Get API Token**

1. Log in to IBM Quantum Platform
2. Click your profile icon → "Account Settings"
3. Copy your API token

**Step 4.3: Configure Token**

**Option A: Environment Variable (Recommended)**
```bash
# Windows (PowerShell)
$env:IBM_QUANTUM_TOKEN="your_token_here"

# Windows (CMD)
set IBM_QUANTUM_TOKEN=your_token_here

# Linux/Mac
export IBM_QUANTUM_TOKEN="your_token_here"
```

**Option B: Configuration File**

Edit `configs/config_ibm_hardware.yaml`:
```yaml
quantum_backend:
  ibm_token: "your_token_here"
```

**Step 4.4: Check Available Backends**

```python
from qiskit_ibm_runtime import QiskitRuntimeService

service = QiskitRuntimeService(channel="ibm_quantum", token="YOUR_TOKEN")
backends = service.backends()

for backend in backends:
    print(f"{backend.name}: {backend.status().pending_jobs} pending jobs")
```

## 🚀 Running the Pipeline

### Scenario 1: Quick Test (Simulator Only)

Run all models on local quantum simulator:

```bash
python run_all_models.py --config configs/config.yaml
```

**Expected output:**
- Training logs for all 5 models
- Confusion matrices and ROC curves
- Comprehensive results comparison
- Quantum advantage report

**Execution time:** 5-15 minutes (depending on dataset size)

### Scenario 2: Aer Simulator (Realistic Noise)

Edit `configs/config.yaml`:
```yaml
quantum_backend:
  backend_type: "aer"
  shots: 1024
```

Run:
```bash
python run_all_models.py --config configs/config.yaml
```

### Scenario 3: IBM Quantum Hardware

**Important:** Real hardware has limited qubits and queue times!

```bash
python run_all_models.py --config configs/config_ibm_hardware.yaml
```

**Notes:**
- Execution time: 30 minutes to several hours (queue dependent)
- Use 4-6 features maximum (`top_k_corr_features: 4`)
- Reduce circuit depth (`reps_feature_map: 1`, `reps_ansatz: 1`)
- Check backend availability first

### Scenario 4: Classical Models Only

Edit `configs/config.yaml`:
```yaml
models_to_run:
  logistic_regression: true
  isolation_forest: true
  xgboost: true
  quantum_vqc: false
  quantum_kernel: false
```

Run:
```bash
python run_all_models.py --config configs/config.yaml
```

**Execution time:** 1-3 minutes

### Scenario 5: Quantum Models Only

Edit `configs/config.yaml`:
```yaml
models_to_run:
  logistic_regression: false
  isolation_forest: false
  xgboost: false
  quantum_vqc: true
  quantum_kernel: true
```

## 📊 Understanding Results

After execution, check the `results/` directory:

### 1. Metrics Comparison (`metrics_comparison.png`)

Visual comparison of all models across 4 metrics:
- **Accuracy**: Overall correctness
- **Precision**: Fraud detection accuracy (minimize false positives)
- **Recall**: Fraud detection coverage (minimize false negatives)
- **F1-Score**: Harmonic mean of precision and recall

**Color coding:**
- 🔵 Teal bars = Classical models
- 🔴 Red bars = Quantum models

### 2. Metrics Table (`metrics_table.csv`)

Numerical results for all models, sorted by F1-score.

Example:
```csv
,accuracy,precision,recall,f1,roc_auc
XGBoost,0.9654,0.8234,0.7891,0.8059,0.9432
Quantum VQC,0.9621,0.8012,0.7654,0.7829,0.9301
Logistic Regression,0.9543,0.7654,0.7234,0.7438,0.9123
```

### 3. Training Time Comparison (`training_time_comparison.png`)

Horizontal bar chart showing training time for each model.

**Typical results:**
- Logistic Regression: 1-5 seconds
- Isolation Forest: 2-10 seconds
- XGBoost: 5-30 seconds
- Quantum VQC: 30-300 seconds (simulator), hours (hardware)
- Quantum Kernel: 60-600 seconds (simulator), hours (hardware)

### 4. Quantum Advantage Report (`quantum_advantage_report.txt`)

Comprehensive text report including:
- Individual model performance
- Best models by metric
- Quantum vs classical comparison
- Improvement percentages
- Conclusions and recommendations

### 5. Individual Visualizations

For each model:
- `confusion_[model].png`: Confusion matrix
- `roc_[model].png`: ROC curve (if applicable)

## 🔍 Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'qiskit'"

**Solution:**
```bash
pip install -r requirements.txt
```

### Issue: "IBM Quantum token required"

**Solution:**
Set environment variable or update config file with your token.

### Issue: "Feature dimension mismatch"

**Solution:**
Quantum models require consistent feature count. Ensure `top_k_corr_features` matches across runs.

### Issue: "Out of memory"

**Solution:**
- Reduce dataset size (sample fewer transactions)
- Reduce `top_k_corr_features` (use 4-6 features)
- Reduce quantum circuit depth (`reps_feature_map: 1`, `reps_ansatz: 1`)

### Issue: "Quantum training is very slow"

**Solution:**
- Reduce `optimizer_maxiter` (try 20-30)
- Use fewer features (`top_k_corr_features: 4`)
- Use smaller training set
- Consider using Aer simulator instead of basic simulator

### Issue: "IBM backend queue is too long"

**Solution:**
- Check backend status: `backend.status().pending_jobs`
- Try different backend (e.g., `ibmq_qasm_simulator` for testing)
- Run during off-peak hours
- Use simulator for development, hardware for final validation

## 🎯 Optimization Tips

### For Faster Execution

1. **Reduce dataset size:**
   ```yaml
   preprocessing:
     test_size: 0.3  # Use more data for testing, less for training
   ```

2. **Reduce features:**
   ```yaml
   preprocessing:
     top_k_corr_features: 4  # Minimum for quantum models
   ```

3. **Reduce quantum iterations:**
   ```yaml
   quantum_vqc:
     optimizer_maxiter: 20
   ```

### For Better Accuracy

1. **Increase features (classical models):**
   ```yaml
   preprocessing:
     top_k_corr_features: 15
   ```

2. **Tune XGBoost:**
   ```yaml
   xgboost:
     n_estimators: 200
     max_depth: 8
     learning_rate: 0.05
   ```

3. **Increase quantum circuit depth:**
   ```yaml
   quantum_vqc:
     reps_feature_map: 3
     reps_ansatz: 3
     optimizer_maxiter: 100
   ```

### For Real Hardware

1. **Minimize circuit depth:**
   ```yaml
   quantum_vqc:
     reps_feature_map: 1
     reps_ansatz: 1
   ```

2. **Use maximum optimization:**
   ```yaml
   quantum_backend:
     optimization_level: 3
   ```

3. **Increase shots for better statistics:**
   ```yaml
   quantum_backend:
     shots: 2048
   ```

## 📈 Expected Performance

### Classical Models (Typical Results)

- **Logistic Regression**: F1 ~0.65-0.75
- **Isolation Forest**: F1 ~0.55-0.65 (unsupervised)
- **XGBoost**: F1 ~0.75-0.85 (best classical)

### Quantum Models (Typical Results)

- **Quantum VQC**: F1 ~0.60-0.75 (simulator)
- **Quantum Kernel**: F1 ~0.65-0.78 (simulator)

**Note:** Results vary based on:
- Dataset size and quality
- Feature selection
- Hyperparameter tuning
- Backend (simulator vs hardware)
- Quantum noise levels

## 🔬 Research Considerations

### When Quantum Shows Advantage

- Small feature spaces (4-8 features)
- Complex non-linear patterns
- High-dimensional kernel spaces
- Specific fraud patterns quantum circuits can capture

### When Classical Performs Better

- Large feature spaces (>10 features)
- Large training datasets
- Well-tuned hyperparameters
- Current NISQ hardware limitations

### Key Insights to Report

1. **Performance Gap**: Quantum vs classical F1-score difference
2. **Training Efficiency**: Time vs accuracy trade-off
3. **Scalability**: How performance changes with features/samples
4. **Hardware Impact**: Simulator vs real hardware comparison
5. **Quantum Advantage**: Specific scenarios where quantum excels

## 📝 Next Steps

1. **Experiment with hyperparameters** in `config.yaml`
2. **Try different feature counts** (4, 6, 8, 10)
3. **Test on IBM hardware** for real quantum results
4. **Analyze quantum advantage report** for insights
5. **Document findings** for research/presentation

## 🆘 Support

- **Qiskit Issues**: [Qiskit GitHub](https://github.com/Qiskit/qiskit)
- **IBM Quantum**: [IBM Quantum Support](https://quantum.ibm.com/support)
- **Project Issues**: Check logs in `fraud_detection_pipeline.log`

---

**Happy Quantum Computing! 🚀**
