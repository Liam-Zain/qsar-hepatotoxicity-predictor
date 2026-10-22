import joblib
import numpy as np
import pandas as pd
from .features import get_morgan_fingerprint

def load_model(model_path: str):
    """Loads the trained model from a file."""
    return joblib.load(model_path)

def predict_hepatotoxicity(smile: str, model_path: str) -> pd.DataFrame:
    """Predicts hepatotoxicity for a list of SMILES."""
    
    # Load the trained model
    model = load_model(model_path)

    # Feature engineering (Fingerprint generation)
    X_fp, img = get_morgan_fingerprint(smile)

    # Predicting
    prediction = model.predict(X_fp)
    probability = np.round((model.predict_proba(X_fp)[:, 1] * 100), 2)
    print("Prediction made successfully...")

    # Outputting the results
    results = pd.DataFrame({
        "SMILES": smile,
        "Molecule": img.show(),
        "Prediction": prediction,
        "Probability (%)": probability
    })

    return results