"""
REST API for Quantum Fraud Detection Service.
Exposes the inference engine via FastAPI.

Run with: uvicorn src.api:app --reload
"""
import os
import sys
import logging
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Add project root to path
sys.path.append(os.getcwd())

from src.inference import FraudInference

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("API")

# Initialize App
app = FastAPI(
    title="Quantum Fraud Detection API",
    description="Hybrid Classical-Quantum Fraud Detection Service",
    version="1.0.0"
)

# Initialize Inference Engine
# We assume artifacts are present in results/
ARTIFACTS_PATH = "results/artifacts/preprocess_artifacts.joblib"
MODELS_DIR = "results/models"
BACKGROUND_DATA = "data/train_transaction.csv"

inference_engine = None

@app.on_event("startup")
def load_model():
    global inference_engine
    try:
        logger.info("Loading Inference Engine...")
        inference_engine = FraudInference(
            artifacts_path=ARTIFACTS_PATH,
            models_dir=MODELS_DIR,
            background_data_path=BACKGROUND_DATA
        )
        logger.info("Inference Engine Loaded Successfully.")
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        # We don't crash app, but health check will fail
        pass

# --- Pydantic Schemas ---
class TransactionInput(BaseModel):
    TransactionID: str
    TransactionAmt: float
    ProductCD: Optional[str] = "W"
    card1: Optional[int] = 1000
    card2: Optional[int] = 555
    card3: Optional[int] = 150
    card4: Optional[str] = "visa"
    card5: Optional[int] = 226
    card6: Optional[str] = "credit"
    addr1: Optional[int] = 300
    addr2: Optional[int] = 87
    dist1: Optional[float] = 10.0
    P_emaildomain: Optional[str] = "gmail.com"
    R_emaildomain: Optional[str] = "gmail.com"
    TransactionDT: Optional[int] = 86400
    # Additional fields can be added as per dataset columns

class FraudPrediction(BaseModel):
    transaction_id: str
    risk_score: float
    is_fraud: bool
    threshold_used: float
    model_used: str
    details: dict

@app.get("/health")
def health_check():
    if inference_engine and inference_engine.models:
        return {"status": "online", "models": list(inference_engine.models.keys())}
    else:
        raise HTTPException(status_code=503, detail="Models not loaded")

@app.post("/predict", response_model=FraudPrediction)
def predict_fraud(transaction: TransactionInput):
    if not inference_engine:
        raise HTTPException(status_code=503, detail="Service unavailable")
    
    try:
        # Convert Pydantic to dict
        data = transaction.dict()
        tx_id = data.pop("TransactionID")
        
        # Run Inference
        result = inference_engine.predict(data)
        
        if "error" in result:
             raise HTTPException(status_code=500, detail=result["error"])
        
        # Prioritize Quantum result if available, else XGBoost
        if "quantum_prob" in result:
            score = result["quantum_prob"]
            pred = result.get("quantum_pred", 0)
            thresh = result.get("quantum_threshold", 0.5)
            model_name = "Quantum VQC"
        else:
            score = result.get("xgboost_prob", 0.0)
            pred = result.get("xgboost_pred", 0)
            thresh = result.get("xgboost_threshold", 0.5)
            model_name = "XGBoost (Classical)"
            
        return FraudPrediction(
            transaction_id=tx_id,
            risk_score=score,
            is_fraud=bool(pred),
            threshold_used=thresh,
            model_used=model_name,
            details=result
        )

    except Exception as e:
        logger.error(f"Prediction Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
