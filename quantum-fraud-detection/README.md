# Quantum Fraud Detection

A comprehensive hybrid quantum-classical machine learning pipeline for fraud detection, comparing classical ML models with quantum algorithms on both simulators and real IBM Quantum hardware.

## 🎯 Project Goal

Leverage **quantum machine learning** to demonstrate quantum advantage over classical computation in fraud detection by:
- Running **3 classical ML models**: Logistic Regression, Isolation Forest, XGBoost
- Running **2 quantum algorithms**: Variational Quantum Classifier (VQC), Quantum Kernel
- Testing on **quantum simulators** and **real IBM Quantum hardware**
- Providing comprehensive performance comparison and analysis

## 📁 Project Structure

```
quantum-fraud-detection/
├── src/                         # Core source code
│   ├── data_loader.py           # Data loading utilities
│   ├── preprocessing.py          # Feature engineering & preprocessing
│   ├── model_classical.py        # Classical ML models (LR, IF, XGBoost)
│   ├── model_quantum.py          # Quantum models (VQC, Quantum Kernel)
│   ├── quantum_backend.py        # Backend management (simulator/IBM hardware)
│   ├── evaluation.py             # Metrics & visualization
│   ├── results_comparison.py     # Comprehensive results analysis
│   └── __init__.py
├── configs/                      # Configuration files
│   ├── config.yaml               # Main configuration
│   └── env_template.txt          # IBM Quantum token template
├── docs/                         # Documentation
│   ├── README.md                 # Documentation index
│   ├── GETTING_STARTED.md        # Setup guide
│   ├── QUICK_START.md            # Fast start guide
│   └── ... (see docs/README.md for full list)
├── notebooks/                    # Jupyter notebooks
│   ├── newfraud.ipynb           # Exploratory analysis
│   └── IBMQiskit.ipynb          # IBM Quantum experiments
├── tests/                        # Test scripts
│   ├── test_feature_selection.py
│   └── test_pipeline.py
├── results/                      # Output directory
│   ├── figures/                 # Visualizations & plots
│   └── logs/                    # Training logs
├── logs/                         # Pipeline logs
├── data/                         # Dataset directory
├── run_all_models.py             # Main pipeline script
├── run.py                        # Alternative runner
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
cd quantum-fraud-detection

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Setup

Place the IEEE-CIS fraud detection dataset CSVs in the `data/` directory:
- `data/train_transaction.csv`
- `data/train_identity.csv`

Or update paths in `configs/config.yaml`.

### 3. Run the Pipeline

**Option A: Fast Prototyping (RECOMMENDED - 2-5 minutes)**
```bash
# Default config uses 10,000 rows for fast testing
python run_all_models.py --config configs/config.yaml
```

**Option B: Full Dataset (30-60 minutes)**
```bash
# Edit configs/config.yaml and set: nrows: null
python run_all_models.py --config configs/config.yaml
```

> 💡 **Tip**: See [PROTOTYPING_GUIDE.md](PROTOTYPING_GUIDE.md) for optimization strategies

**Option C: Run on IBM Quantum Hardware**

1. Get your IBM Quantum token from [IBM Quantum Platform](https://quantum.ibm.com/)
2. Update `configs/config.yaml`:
   ```yaml
   quantum_backend:
     backend_type: "ibm_quantum"
     ibm_token: "YOUR_IBM_QUANTUM_TOKEN"
     ibm_backend_name: "ibm_brisbane"  # or other available backend
   ```
3. Run the pipeline:
   ```bash
   python run_all_models.py --config configs/config.yaml
   ```

## 🔧 Configuration

Edit `configs/config.yaml` to customize:

### Classical Models
- **Logistic Regression**: Regularization, oversampling
- **Isolation Forest**: Contamination rate, estimators
- **XGBoost**: Tree depth, learning rate, GPU acceleration

### Quantum Models
- **VQC**: Feature map repetitions, ansatz depth, optimizer iterations
- **Quantum Kernel**: Feature map, SVM parameters

### Backend Selection
- `simulator`: Local ideal simulator (fast)
- `aer`: Qiskit Aer simulator (realistic noise)
- `ibm_quantum`: Real IBM Quantum hardware

### Model Selection
Enable/disable specific models:
```yaml
models_to_run:
  logistic_regression: true
  isolation_forest: true
  xgboost: true
  quantum_vqc: true
  quantum_kernel: true
```

## 📊 Results & Analysis

After running the pipeline, check the `results/` directory for:

1. **Metrics Comparison** (`metrics_comparison.png`)
   - Bar charts comparing accuracy, precision, recall, F1-score

2. **Metrics Table** (`metrics_table.csv`)
   - Detailed numerical results for all models

3. **Training Time Comparison** (`training_time_comparison.png`)
   - Performance overhead analysis

4. **Quantum Advantage Report** (`quantum_advantage_report.txt`)
   - Comprehensive analysis of quantum vs classical performance
   - Improvement percentages
   - Best model recommendations

5. **Individual Model Visualizations**
   - Confusion matrices for each model
   - ROC curves (where applicable)

## 🧪 Models Overview

### Classical Models

1. **Logistic Regression**
   - Linear baseline with L2 regularization
   - Optional SMOTE oversampling for imbalanced data

2. **Isolation Forest**
   - Unsupervised anomaly detection
   - Effective for outlier-based fraud detection

3. **XGBoost**
   - Gradient boosting ensemble
   - State-of-the-art classical performance

### Quantum Models

1. **Variational Quantum Classifier (VQC)**
   - Parameterized quantum circuit
   - ZZ feature map + TwoLocal ansatz
   - COBYLA optimizer

2. **Quantum Kernel**
   - Quantum kernel-based SVM
   - Fidelity-based quantum kernel
   - Exploits quantum feature space

## 🔬 IBM Quantum Hardware

To run on real quantum hardware:

1. **Get IBM Quantum Access**
   - Sign up at [IBM Quantum](https://quantum.ibm.com/)
   - Copy your API token

2. **Select Backend**
   - Available backends: `ibm_brisbane`, `ibm_kyoto`, `ibmq_qasm_simulator`
   - Check [IBM Quantum Services](https://quantum.ibm.com/services) for availability

3. **Configure**
   ```yaml
   quantum_backend:
     backend_type: "ibm_quantum"
     ibm_token: "YOUR_TOKEN"
     ibm_backend_name: "ibm_brisbane"
     shots: 1024
     optimization_level: 1
   ```

4. **Note**: Real hardware execution may take longer due to queue times

## 📈 Expected Outcomes

The pipeline will demonstrate:

✅ **Performance Comparison**: Classical vs Quantum models on fraud detection  
✅ **Quantum Advantage Analysis**: Where quantum models excel  
✅ **Scalability Insights**: Training time vs accuracy trade-offs  
✅ **Hardware Validation**: Simulator vs real quantum hardware results  

## 🛠️ Advanced Usage

### Custom Feature Selection
Modify `top_k_corr_features` in `config.yaml` to adjust feature count for quantum models (recommended: 4-8 features for current quantum hardware).

### Hyperparameter Tuning
Each model configuration can be tuned independently in `config.yaml`.

### Adding New Models
Extend `src/model_classical.py` or `src/model_quantum.py` with new model implementations.

## 📝 Citation

If you use this project, please cite:
```
Quantum Fraud Detection: A Hybrid Quantum-Classical Machine Learning Approach
IEEE-CIS Fraud Detection Dataset
```

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## 📄 License

MIT License - See LICENSE file for details

## 📚 Documentation

For detailed guides and documentation, see the **[docs/](docs/)** directory:
- [Getting Started Guide](docs/GETTING_STARTED.md)
- [Quick Start](docs/QUICK_START.md)
- [Prototyping Guide](docs/PROTOTYPING_GUIDE.md)
- [Feature Selection Guide](docs/FEATURE_SELECTION_GUIDE.md)
- [Results Interpretation](docs/RESULTS_INTERPRETATION.md)
- [Full Documentation Index](docs/README.md)

## 🔗 Resources

- [Qiskit Documentation](https://qiskit.org/documentation/)
- [Qiskit Machine Learning](https://qiskit.org/ecosystem/machine-learning/)
- [IBM Quantum Platform](https://quantum.ibm.com/)
- [IEEE-CIS Fraud Detection Dataset](https://www.kaggle.com/c/ieee-fraud-detection)

---

**Note**: Quantum models require significant computational resources. Start with small feature sets (4-8 features) and increase gradually.
