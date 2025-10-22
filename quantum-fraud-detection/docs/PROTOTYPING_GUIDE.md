# Prototyping Guide - Quick Testing Configurations

This guide helps you optimize the pipeline for fast prototyping and testing.

## 🚀 Quick Start Configurations

### Ultra-Fast Prototyping (1-2 minutes)
**Best for**: Initial testing, debugging, quick iterations

```yaml
data:
  nrows: 5000

preprocessing:
  top_k_corr_features: 4

quantum_vqc:
  reps_feature_map: 1
  reps_ansatz: 1
  optimizer_maxiter: 20

quantum_kernel:
  reps_feature_map: 1

models_to_run:
  logistic_regression: true
  isolation_forest: false  # Disable for speed
  xgboost: true
  quantum_vqc: true
  quantum_kernel: false  # Disable for speed
```

**Expected time**: ~1-2 minutes  
**Trade-off**: Lower accuracy, but fast feedback

---

### Fast Prototyping (2-5 minutes) ⭐ **RECOMMENDED**
**Best for**: Regular development, feature testing

```yaml
data:
  nrows: 10000  # Current default

preprocessing:
  top_k_corr_features: 4

quantum_vqc:
  reps_feature_map: 2
  reps_ansatz: 2
  optimizer_maxiter: 50

quantum_kernel:
  reps_feature_map: 2

models_to_run:
  logistic_regression: true
  isolation_forest: true
  xgboost: true
  quantum_vqc: true
  quantum_kernel: true
```

**Expected time**: ~2-5 minutes  
**Trade-off**: Good balance between speed and accuracy

---

### Medium Testing (5-15 minutes)
**Best for**: Pre-production validation, hyperparameter tuning

```yaml
data:
  nrows: 50000

preprocessing:
  top_k_corr_features: 6

quantum_vqc:
  reps_feature_map: 2
  reps_ansatz: 2
  optimizer_maxiter: 100

quantum_kernel:
  reps_feature_map: 2

models_to_run:
  logistic_regression: true
  isolation_forest: true
  xgboost: true
  quantum_vqc: true
  quantum_kernel: true
```

**Expected time**: ~5-15 minutes  
**Trade-off**: More representative results

---

### Full Production (30-60 minutes)
**Best for**: Final results, publication, benchmarking

```yaml
data:
  nrows: null  # Load all ~590k rows

preprocessing:
  top_k_corr_features: 8

quantum_vqc:
  reps_feature_map: 2
  reps_ansatz: 2
  optimizer_maxiter: 100

quantum_kernel:
  reps_feature_map: 2

models_to_run:
  logistic_regression: true
  isolation_forest: true
  xgboost: true
  quantum_vqc: true
  quantum_kernel: true
```

**Expected time**: ~30-60 minutes  
**Trade-off**: Best accuracy, longest runtime

---

## 📊 Performance Comparison

| Configuration | Rows | Features | VQC Iters | Time | Accuracy* |
|--------------|------|----------|-----------|------|-----------|
| Ultra-Fast   | 5k   | 4        | 20        | 1-2m | ~70-75%   |
| Fast         | 10k  | 4        | 50        | 2-5m | ~72-78%   |
| Medium       | 50k  | 6        | 100       | 5-15m| ~75-82%   |
| Full         | 590k | 8        | 100       | 30-60m| ~78-85%  |

*Approximate F1-scores for best model (typically XGBoost)

---

## 🎯 Model-Specific Optimizations

### Classical Models Only (Fastest)
Disable quantum models for ultra-fast classical baseline:

```yaml
models_to_run:
  logistic_regression: true
  isolation_forest: true
  xgboost: true
  quantum_vqc: false
  quantum_kernel: false
```

**Time**: ~30 seconds - 2 minutes

---

### Quantum Models Only
Focus on quantum performance:

```yaml
models_to_run:
  logistic_regression: false
  isolation_forest: false
  xgboost: false
  quantum_vqc: true
  quantum_kernel: true
```

**Time**: ~1-3 minutes (with 10k rows)

---

## 💡 Tips for Faster Prototyping

### 1. **Reduce Dataset Size**
```yaml
data:
  nrows: 5000  # Start small, increase gradually
```

### 2. **Limit Features**
```yaml
preprocessing:
  top_k_corr_features: 4  # Minimum for quantum models
```

### 3. **Reduce Quantum Circuit Complexity**
```yaml
quantum_vqc:
  reps_feature_map: 1  # Reduce from 2
  reps_ansatz: 1       # Reduce from 2
  optimizer_maxiter: 20 # Reduce from 50
```

### 4. **Disable Slow Models**
```yaml
models_to_run:
  isolation_forest: false  # Can be slow on large datasets
  quantum_kernel: false    # Slower than VQC
```

### 5. **Use Simulator (Not Hardware)**
```yaml
quantum_backend:
  backend_type: "simulator"  # Much faster than real hardware
```

---

## 🔄 Workflow Recommendations

### Development Cycle
1. **Start**: Ultra-Fast (5k rows, 4 features, 20 iters)
2. **Debug**: Fast (10k rows, 4 features, 50 iters)
3. **Validate**: Medium (50k rows, 6 features, 100 iters)
4. **Production**: Full (590k rows, 8 features, 100 iters)

### Quick Testing Command
```bash
# Edit config.yaml to set nrows: 5000
python run_all_models.py --config configs/config.yaml
```

---

## 📈 Expected Results by Configuration

### Ultra-Fast (5k rows)
- **Load time**: ~2 seconds
- **Preprocessing**: ~5 seconds
- **Classical models**: ~10 seconds
- **Quantum models**: ~60 seconds
- **Total**: ~1-2 minutes

### Fast (10k rows) - RECOMMENDED
- **Load time**: ~3 seconds
- **Preprocessing**: ~8 seconds
- **Classical models**: ~15 seconds
- **Quantum models**: ~120 seconds
- **Total**: ~2-5 minutes

### Medium (50k rows)
- **Load time**: ~10 seconds
- **Preprocessing**: ~30 seconds
- **Classical models**: ~45 seconds
- **Quantum models**: ~300 seconds
- **Total**: ~5-15 minutes

### Full (590k rows)
- **Load time**: ~45 seconds
- **Preprocessing**: ~120 seconds
- **Classical models**: ~180 seconds
- **Quantum models**: ~600 seconds
- **Total**: ~30-60 minutes

---

## 🎓 When to Use Each Configuration

| Use Case | Configuration | Reason |
|----------|--------------|--------|
| Debugging code | Ultra-Fast | Fastest feedback loop |
| Testing new features | Fast | Good balance |
| Hyperparameter tuning | Medium | More representative |
| Final benchmarking | Full | Best accuracy |
| Demo/presentation | Fast or Medium | Quick but impressive |
| Research paper | Full | Publication-quality |

---

## 🚨 Common Issues

### Issue: "Out of Memory"
**Solution**: Reduce `nrows` to 5000 or 10000

### Issue: "Quantum training too slow"
**Solution**: 
- Reduce `optimizer_maxiter` to 20-30
- Reduce `reps_feature_map` and `reps_ansatz` to 1
- Set `quantum_vqc: false` temporarily

### Issue: "Results not representative"
**Solution**: Increase to at least 50k rows for validation

---

## ✅ Current Configuration Status

Your current `config.yaml` is set to:
- **nrows**: 10000 (Fast prototyping mode)
- **features**: 4 (Optimal for quantum)
- **VQC iterations**: 50 (Balanced)

This is the **recommended configuration** for regular development! 🎯

To switch modes, simply edit the `nrows` value in `configs/config.yaml`.
