# Getting Started with Quantum Fraud Detection

**Welcome!** This guide will get you up and running in 15 minutes.

## 🎯 What You'll Achieve

By the end of this guide, you'll have:
- ✅ A working quantum fraud detection pipeline
- ✅ Results comparing 3 classical + 2 quantum models
- ✅ Comprehensive performance analysis
- ✅ Understanding of quantum advantage

## ⚡ Quick Start (3 Steps)

### Step 1: Install Dependencies (2 minutes)

```bash
cd quantum-fraud-detection
pip install -r requirements.txt
```

**Expected output:**
```
Successfully installed qiskit-1.4.4 xgboost-2.0.0 ...
```

### Step 2: Prepare Data (5 minutes)

**Option A: Use Sample Data (Fastest)**
```python
# Create a small sample for testing
import pandas as pd

# Download from Kaggle and sample
df_txn = pd.read_csv('path/to/train_transaction.csv')
df_id = pd.read_csv('path/to/train_identity.csv')

# Sample 5000 transactions
df_txn_sample = df_txn.sample(n=5000, random_state=42)
df_id_sample = df_id[df_id['TransactionID'].isin(df_txn_sample['TransactionID'])]

# Save to data directory
df_txn_sample.to_csv('data/train_transaction.csv', index=False)
df_id_sample.to_csv('data/train_identity.csv', index=False)
```

**Option B: Use Full Dataset**
1. Download from [Kaggle IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection/data)
2. Place `train_transaction.csv` and `train_identity.csv` in `data/` folder

### Step 3: Run Pipeline (5-10 minutes)

```bash
python run_all_models.py --config configs/config.yaml
```

**What happens:**
1. ⏳ Loads and preprocesses data
2. 🔵 Trains 3 classical models (Logistic Regression, Isolation Forest, XGBoost)
3. 🔴 Trains 2 quantum models (VQC, Quantum Kernel)
4. 📊 Generates comprehensive comparison
5. ✅ Saves results to `results/` directory

**Expected output:**
```
Starting Quantum Fraud Detection Pipeline
Loading and preprocessing data...
Training samples: 4000, Test samples: 1000
Features: 8

Training Logistic Regression...
Training completed in 2.34 seconds
Logistic Regression Metrics: {'accuracy': 0.9543, 'f1': 0.7438, ...}

Training Isolation Forest...
Training completed in 3.12 seconds
...

Training XGBoost...
Training completed in 8.45 seconds
...

Training Quantum VQC...
⚠️ This may take several minutes...
Training completed in 145.67 seconds
...

Training Quantum Kernel...
⚠️ This may take several minutes...
Training completed in 234.89 seconds
...

Generating comprehensive results comparison...
All results saved to: results

Pipeline completed successfully!
```

## 📊 View Your Results

### 1. Quick Summary
```bash
cat results/metrics_table.csv
```

Example output:
```csv
,accuracy,precision,recall,f1,roc_auc
XGBoost,0.9654,0.8234,0.7891,0.8059,0.9432
Quantum Kernel,0.9621,0.8012,0.7654,0.7829,0.9301
Logistic Regression,0.9543,0.7654,0.7234,0.7438,0.9123
Quantum VQC,0.9512,0.7456,0.7012,0.7228,0.9087
Isolation Forest,0.9234,0.6543,0.6123,0.6325,0.8765
```

### 2. Quantum Advantage Report
```bash
cat results/quantum_advantage_report.txt
```

### 3. Visualizations
Open these files in `results/`:
- `metrics_comparison.png` - Performance comparison chart
- `training_time_comparison.png` - Training time analysis
- `figures/confusion_*.png` - Confusion matrices for each model
- `figures/roc_*.png` - ROC curves

## 🎓 Understanding Your Results

### Key Metrics Explained

**F1-Score** (Primary Metric)
- **0.75-0.85**: Excellent performance ⭐⭐⭐
- **0.65-0.75**: Good performance ⭐⭐
- **0.55-0.65**: Moderate performance ⭐
- **<0.55**: Needs improvement

**What to Look For:**
1. **Best Classical Model**: Usually XGBoost
2. **Best Quantum Model**: Usually Quantum Kernel
3. **Quantum Advantage**: Compare best quantum vs best classical

### Example Interpretation

**Scenario 1: Quantum Advantage**
```
Best Classical F1: 0.8059
Best Quantum F1: 0.8234
Improvement: +2.17%
```
✅ **Conclusion**: Quantum models show advantage!

**Scenario 2: Comparable Performance**
```
Best Classical F1: 0.8059
Best Quantum F1: 0.7829
Improvement: -2.85%
```
⚖️ **Conclusion**: Quantum models are competitive, showing promise for future hardware.

**Scenario 3: Classical Better**
```
Best Classical F1: 0.8059
Best Quantum F1: 0.7228
Improvement: -10.31%
```
📊 **Conclusion**: Classical models currently better, but quantum shows potential with optimization.

## 🔧 Customization

### Run Faster (Classical Only)

Edit `configs/config.yaml`:
```yaml
models_to_run:
  logistic_regression: true
  isolation_forest: true
  xgboost: true
  quantum_vqc: false  # Disable
  quantum_kernel: false  # Disable
```

Run time: **1-3 minutes**

### Adjust Feature Count

Edit `configs/config.yaml`:
```yaml
preprocessing:
  top_k_corr_features: 4  # Try 4, 6, 8, or 10
```

**Recommendation:**
- **4-6 features**: Best for quantum models
- **8-10 features**: Better for classical models
- **>10 features**: Quantum models may struggle

### Tune Quantum Models

Edit `configs/config.yaml`:
```yaml
quantum_vqc:
  reps_feature_map: 1  # Reduce for faster training
  reps_ansatz: 1       # Reduce for faster training
  optimizer_maxiter: 20  # Reduce for faster training
```

## 🚀 Next Steps

### Beginner
1. ✅ Run with default configuration
2. 📊 Review results
3. 📖 Read quantum advantage report
4. 🎓 Understand metrics

### Intermediate
1. 🔧 Experiment with feature counts
2. ⚙️ Tune hyperparameters
3. 📈 Analyze confusion matrices
4. 🔬 Compare different configurations

### Advanced
1. 🌐 Run on IBM Quantum hardware
2. 🧪 Implement custom models
3. 📊 Statistical significance testing
4. 📝 Write research paper

## 🔬 Running on IBM Quantum Hardware

### Prerequisites
1. Create account at [IBM Quantum](https://quantum.ibm.com/)
2. Get API token from account settings

### Configuration

**Option 1: Environment Variable (Recommended)**
```bash
# Windows PowerShell
$env:IBM_QUANTUM_TOKEN="your_token_here"

# Linux/Mac
export IBM_QUANTUM_TOKEN="your_token_here"
```

**Option 2: Config File**

Edit `configs/config_ibm_hardware.yaml`:
```yaml
quantum_backend:
  backend_type: "ibm_quantum"
  ibm_token: "your_token_here"
  ibm_backend_name: "ibm_brisbane"
```

### Run on Hardware
```bash
python run_all_models.py --config configs/config_ibm_hardware.yaml
```

**⚠️ Important:**
- Execution time: 30 min - several hours (queue dependent)
- Use 4-6 features maximum
- Reduce circuit depth for hardware constraints
- Check backend availability first

## 📚 Interactive Tutorial

For step-by-step learning, use the Jupyter notebook:

```bash
jupyter notebook notebooks/quick_start.ipynb
```

**What's included:**
- Interactive code cells
- Detailed explanations
- Visualization examples
- Experimentation ideas

## 🆘 Troubleshooting

### Problem: "ModuleNotFoundError"
**Solution:**
```bash
pip install -r requirements.txt
```

### Problem: "FileNotFoundError: data/train_transaction.csv"
**Solution:**
- Download dataset from Kaggle
- Place files in `data/` directory
- Or update paths in `config.yaml`

### Problem: "Quantum training is very slow"
**Solution:**
- Reduce `optimizer_maxiter` to 20
- Reduce `top_k_corr_features` to 4
- Reduce circuit depth (`reps_feature_map: 1`)

### Problem: "Out of memory"
**Solution:**
- Use smaller dataset sample
- Reduce feature count
- Close other applications

### More Help
See `SETUP_GUIDE.md` for detailed troubleshooting.

## 📖 Documentation

- **README.md**: Project overview
- **SETUP_GUIDE.md**: Detailed setup instructions
- **RESULTS_INTERPRETATION.md**: How to interpret results
- **PROJECT_SUMMARY.md**: Comprehensive technical overview
- **CHECKLIST.md**: Verification checklist

## 🎯 Success Checklist

- [ ] Dependencies installed
- [ ] Data prepared
- [ ] Pipeline runs successfully
- [ ] Results generated
- [ ] Metrics reviewed
- [ ] Quantum advantage report read
- [ ] Key findings identified

**All checked?** 🎉 Congratulations! You're ready to explore quantum machine learning!

## 💡 Tips for Best Results

1. **Start Small**: Use 5,000 samples for initial testing
2. **Use Simulator First**: Test on simulator before hardware
3. **Optimize Features**: 4-6 features work best for quantum
4. **Compare Fairly**: Same features for all models
5. **Document Everything**: Save configurations and results
6. **Iterate**: Try different hyperparameters

## 🌟 What Makes This Special

✨ **Comprehensive**: 5 models in one pipeline  
✨ **Automated**: One command runs everything  
✨ **Flexible**: Easy configuration via YAML  
✨ **Production-Ready**: IBM Quantum hardware support  
✨ **Research-Grade**: Publication-ready results  
✨ **Educational**: Learn classical and quantum ML  

## 🚀 Ready to Start?

```bash
# 1. Install
pip install -r requirements.txt

# 2. Run
python run_all_models.py --config configs/config.yaml

# 3. Explore
cat results/quantum_advantage_report.txt
```

**Happy Quantum Computing! 🎉**

---

**Questions?** Check the documentation or review the code comments.

**Found a bug?** Create an issue with details.

**Want to contribute?** Pull requests welcome!
