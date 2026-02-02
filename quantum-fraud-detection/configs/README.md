# ⚙️ Configuration System Documentation

This project uses a **First-Class Configuration System** validated by strict Pydantic schemas. 
All tunable parameters live here. Code logic is separate from configuration.

## 📂 Configuration Files

| File | Environment | Purpose |
|:--- |:--- |:--- |
| `config.yaml` | **Development / Simulator** | Default config for local dev and testing. Uses `simulator` backend. |
| `config_ibm_hardware.yaml` | **Production / Hardware** | Optimized for real IBM Quantum hardware (`ibm_torino`). **Requires Token.** |
| `config_sanity_check.yaml` | **CI / Safety** | Minimal run (few rows, few shots) to verify pipeline integrity quickly. |
| `env_template.txt` | **Secrets** | Template for `.env`. Never commit actual secrets! |

## 🛡️ Validation & Safety
Configuration is validated at runtime using `src/config_schema.py`.
- **Type Checking**: Ensures integers are integers, booleans are booleans.
- **Range Checking**: `shots`, `probabilities`, and `thresholds` are validated against safe ranges.
- **Backend Safety**: Prevents simulator settings from leaking into hardware runs.

## 🛠️ Key Parameters

### ⚠️ Dangerous Parameters (Tune with Caution)
- `nrows`: Set to `null` for full training (~2 hours). Keep small (e.g., 5000) for debugging.
- `shots`: Higher = More precision but slower/costlier. Default: 1024 or 4096.
- `backend_type`: Switching to `ibm_quantum` determines cost.

### 🧪 Tuning Parameters
- `top_k_features`: Controls dimensionality (Quantum Qubits).
- `models_to_run`: Toggle specific models (e.g., enable `quantum_kernel` only for final runs).

## 🚀 How to Use
Pass the config file argument to the runner:
```bash
# Default (Dev)
python run_all_models.py

# Hardware (Prod)
python run_all_models.py --config configs/config_ibm_hardware.yaml

# Sanity Check
python run_all_models.py --config configs/config_sanity_check.yaml
```
