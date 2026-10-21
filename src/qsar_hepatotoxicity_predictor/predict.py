import joblib
import numpy as np
import pandas as pd
from .features import get_morgan_fingerprint

def load_model(model_path: str):
    """Loads the trained model from a file."""
    return joblib.load(model_path)

def predict_hepatotoxicity(smiles_list: list, model_path: str) -> pd.DataFrame:
    """Predicts hepatotoxicity for a list of SMILES."""
    
    # Load the trained model
    model = load_model(model_path)

    # Feature engineering (Fingerprint generation)
    X_fp = get_morgan_fingerprint(np.array(smiles_list))

    # Predicting
    predictions = model.predict(X_fp)
    probabilities = np.round((model.predict_proba(X_fp)[:, 1] * 100), 2)
    print("Predictions made successfully...")

    # Outputting the results
    results = pd.DataFrame({
        "SMILES": smiles_list,
        "Prediction": predictions,
        "Probability (%)": probabilities
    })

    return results