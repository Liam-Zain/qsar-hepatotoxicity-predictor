import numpy as np
from fastapi import APIRouter
from qsar_hepatotoxicity_predictor.features import get_morgan_fingerprint
import joblib

from qsar_hepatotoxicity_predictor.api.schemas import SMILESRequest, BatchSMILESRequest, PredictionResponse

router = APIRouter()

model = joblib.load("models/hepatotoxicity_model.pkl")

@router.get("/")
def read_root():
    return {"message": "Welcome to the QSAR Hepatotoxicity Predictor API"}

@router.post("/predict/", response_model=PredictionResponse)
def predict_single(request: SMILESRequest):
    """Predicts hepatotoxicity for a single SMILES."""
    fp, img = get_morgan_fingerprint(request.smiles)
    X_fp = fp.reshape(1, -1)

    prediction = int(model.predict(X_fp)[0])
    probability = float(np.round(model.predict_proba(X_fp)[0][1] * 100, 2))

    if probability >= 70:
        likelihood = "Most likely hepatotoxic"
    elif probability >= 50:
        likelihood = "Likely hepatotoxic"
    elif probability >= 30:
        likelihood = "Likely not hepatoxic"
    else:
        likelihood = "Most likely not hepatotoxic"
        
    return PredictionResponse(
        smiles=request.smiles,
        prediction=prediction,
        probability=probability,
        is_hepatotoxic=(prediction == 1),
        likelihood=likelihood
    )

@router.post("/predict/batch", response_model=list[PredictionResponse])
def predict_batch(request: BatchSMILESRequest):
    results = []
    for smile in request.smiles_list:
        try:
            fp, img = get_morgan_fingerprint(smile)
            X_fp = fp.reshape(1, -1)

            prediction = int(model.predict(X_fp)[0])
            probability = float(np.round(model.predict_proba(X_fp)[0][1] * 100, 2))
            
            if probability >= 70:
                likelihood = "Most likely hepatotoxic"
            elif probability >= 50:
                likelihood = "Likely hepatotoxic"
            elif probability >= 30:
                likelihood = "Likely not hepatoxic"
            else:
                likelihood = "Most likely not hepatotoxic"

            results.append(PredictionResponse(
            smiles=smile,
            prediction=prediction,
            probability=probability,
            is_hepatotoxic=(prediction == 1),
            likelihood=likelihood
        ))

        except ValueError as e:
            print("Error processing SMILES:", e)
            results.append(PredictionResponse(
                    smiles=smile,
                    prediction=None,
                    probability=None,
                    is_hepatotoxic=False,
                    likelihood="Not applicable"
                ))
            
    return results
                