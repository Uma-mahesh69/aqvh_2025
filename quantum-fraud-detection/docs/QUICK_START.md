# Quick Start Guide

## Running the Pipeline

```bash
cd D:\quantum_valley\quantum-fraud-detection
python run_all_models.py
```

## What to Expect

### 1. Data Loading (10-15 seconds)
```
Loading data with nrows=10000...
[OK] Loaded 10,000 transaction rows and 10,000 identity rows
```

### 2. Feature Selection (30-60 seconds)
The ensemble method will run 4 different feature selection algorithms:
```
============================================================
ENSEMBLE FEATURE SELECTION
============================================================
Mutual Info - Top 8 features: [...]
RF Importance - Top 8 features: [...]
RFE - Top 8 features: [...]

Voting Results:
  - V114: 2 votes
  - V122: 2 votes
  ...

Final Selection: 8 features
============================================================

FEATURE SELECTION COMPLETE
Method: ensemble
Selected 8 features: ['V114', 'V122', 'V121', 'V123', 'C8', 'V125', 'V303', 'TransactionID']
```

### 3. Model Training & Evaluation
Each model will show:
- Training time
- Accuracy, Precision, Recall, F1-Score
- ROC-AUC score
- Confusion matrix saved to `results/figures/`

### 4. Final Results
Comparison table saved to `results/model_comparison.csv`

## Current Issues & Solutions

### ❌ XGBoost Not Installed
**Error**: `XGBoost is not installed`

**Solution**:
```bash
pip install xgboost
```

### ⚠️ Tkinter Threading Warnings
**Error**: `RuntimeError: main thread is not in main loop`

**Impact**: Non-critical, just matplotlib backend warnings. Plots still save correctly.

**Optional Fix**: Add to top of `run_all_models.py`:
```python
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
```

### ✅ NumPy Deprecation Warning
**Status**: Fixed! Updated to use `.astype(int)` for bincount.

## Expected Performance

With **ensemble feature selection** (8 features):

| Model | Expected Accuracy | Training Time |
|-------|------------------|---------------|
| Logistic Regression | 85-90% | 1-2 sec |
| Isolation Forest | 70-80% | 2-3 sec |
| XGBoost | 90-95% | 5-10 sec |
| Quantum VQC | 80-85% | 30-60 sec |
| Quantum Kernel | 80-85% | 20-40 sec |

## Configuration Options

Edit `configs/config.yaml`:

### Fast Prototyping (Current)
```yaml
nrows: 10000  # ~2-5 min total
feature_selection_method: "ensemble"
top_k_features: 8
```

### Maximum Performance
```yaml
nrows: null  # Full dataset, ~30-60 min total
feature_selection_method: "ensemble"
top_k_features: 12
ensemble_voting_threshold: 3  # More conservative
```

### Quick Testing
```yaml
nrows: 5000  # ~1-2 min total
feature_selection_method: "correlation"  # Fastest method
top_k_features: 6
```

## Troubleshooting

### Pipeline runs but no feature selection output?
- Check that logging is configured correctly
- Verify `configs/config.yaml` has the new parameters
- Run `python test_feature_selection.py` to test feature selection independently

### Out of memory errors?
- Reduce `nrows` to 5000 or 1000
- Reduce `top_k_features` to 6 or 4

### Quantum models taking too long?
- Reduce `reps_feature_map` and `reps_ansatz` to 1
- Reduce `optimizer_maxiter` to 25
- Use fewer features (4-6)

## Next Steps

1. **Install XGBoost**: `pip install xgboost`
2. **Run pipeline**: `python run_all_models.py`
3. **Check results**: Look in `results/` folder for:
   - `model_comparison.csv` - Performance metrics
   - `figures/` - Confusion matrices and ROC curves
   - `logs/` - Detailed logs

4. **Experiment with feature selection**:
   - Try different methods: `correlation`, `mutual_info`, `rf_importance`, `rfe`, `ensemble`
   - Adjust `top_k_features` (6-12 recommended)
   - Change `ensemble_voting_threshold` (2-4)

5. **Scale to full dataset**:
   - Set `nrows: null` in config
   - Expect 30-60 min runtime
   - Better model performance

## Documentation

- `FEATURE_SELECTION_GUIDE.md` - Detailed feature selection documentation
- `PROTOTYPING_GUIDE.md` - Dataset size recommendations
- `README.md` - Project overview
