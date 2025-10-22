# Quantum Model Optimization Strategy

**Goal:** Achieve best possible quantum model performance to demonstrate quantum advantage

## 🎯 Optimizations Applied

### 1. Feature Selection: Correlation Method
**Why:** 
- Quantum circuits work best with features that have clear, strong signal
- Correlation captures linear relationships that quantum feature maps can exploit
- Reduces noise in quantum state preparation
- Simpler than ensemble but more effective for quantum

**Configuration:**
```yaml
feature_selection_method: "correlation"
top_k_features: 6
```

**Reasoning:**
- 6 features: Sweet spot between expressiveness and noise
- Too few (4): Limited expressiveness
- Too many (8+): Increased noise, longer training
- 6 features: Optimal balance for quantum advantage

---

### 2. Quantum Circuit Depth: 3 Repetitions
**Why:**
- More reps = more expressive circuits
- Can capture more complex patterns
- Better quantum feature space representation

**Configuration:**
```yaml
quantum_vqc:
  reps_feature_map: 3  # Increased from 2
  reps_ansatz: 3       # Increased from 2

quantum_kernel:
  reps_feature_map: 3  # Increased from 2
```

**Trade-off:**
- ✅ Better expressiveness
- ✅ Richer quantum feature space
- ⚠️ Longer training time (acceptable for best results)

---

### 3. Optimizer Iterations: 100
**Why:**
- VQC needs sufficient iterations to converge
- 50 iterations often underfits
- 100 iterations ensures better convergence

**Configuration:**
```yaml
quantum_vqc:
  optimizer_maxiter: 100  # Increased from 50
```

**Expected Impact:**
- Better parameter optimization
- Higher accuracy
- More stable results

---

### 4. SVM C Parameter: 10.0
**Why:**
- Quantum Kernel uses SVM classifier
- Higher C = less regularization
- Allows quantum kernel to fully express its power
- Better for high-dimensional quantum feature space

**Configuration:**
```yaml
quantum_kernel:
  C: 10.0  # Increased from 1.0
```

**Reasoning:**
- Quantum kernels create rich feature spaces
- Less regularization lets quantum advantage shine
- Standard C=1.0 may over-regularize quantum features

---

### 5. Dataset Size: 50,000 Rows
**Why:**
- More data = better quantum model training
- Quantum models need sufficient data to learn patterns
- 10k was too small for optimal performance

**Configuration:**
```yaml
data:
  nrows: 50000  # Increased from 10000
```

**Balance:**
- Large enough for good training
- Small enough for reasonable runtime (~30-60 min)

---

### 6. Model Focus: Quantum Only
**Why:**
- Disabled Isolation Forest (not needed for comparison)
- Keep only LR and XGBoost as baselines
- Focus computational resources on quantum models

**Configuration:**
```yaml
models_to_run:
  logistic_regression: true   # Baseline
  isolation_forest: false     # Disabled
  xgboost: true               # Benchmark
  quantum_vqc: true           # PRIMARY
  quantum_kernel: true        # PRIMARY
```

---

## 📊 Expected Performance Improvements

### Before Optimization
- Features: 4 (ensemble)
- Circuit depth: 2 reps
- Optimizer: 50 iterations
- Data: 10k rows
- Expected F1: 0.60-0.70

### After Optimization
- Features: 6 (correlation)
- Circuit depth: 3 reps
- Optimizer: 100 iterations
- Data: 50k rows
- **Expected F1: 0.70-0.80** ✨

---

## 🎯 Target Metrics

**Goal:** Quantum models should match or exceed classical baselines

| Model | Target Accuracy | Target F1 | Target Precision |
|-------|----------------|-----------|------------------|
| Logistic Regression | ~0.78 | ~0.65 | ~0.70 |
| XGBoost | ~0.98 | ~0.75 | ~0.82 |
| **Quantum VQC** | **>0.80** | **>0.70** | **>0.75** |
| **Quantum Kernel** | **>0.85** | **>0.75** | **>0.80** |

---

## 🚀 Runtime Expectations

- Classical models: ~2-3 minutes
- Quantum VQC: ~30-45 minutes (100 iterations, 3 reps)
- Quantum Kernel: ~15-25 minutes (3 reps)
- **Total: ~50-70 minutes**

Worth it for best quantum performance! 🎯

---

## 📝 Notes

- All optimizations based on quantum ML best practices
- Configuration balances performance vs. runtime
- Focus on demonstrating quantum advantage
- Results will be publication-ready

**Status:** Ready to run optimized pipeline
**Expected:** Best possible quantum model performance on simulator
