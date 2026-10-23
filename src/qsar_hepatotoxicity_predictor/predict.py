import joblib
import numpy as np
import pandas as pd
import base64
from io import BytesIO
from .features import get_morgan_fingerprint

def load_model(model_save_path: str):
    """Loads the trained model from a file."""
    return joblib.load(model_save_path)

def predict_hepatotoxicity(smile: str, model_save_path: str) -> pd.DataFrame:
    """Predicts hepatotoxicity for a list of SMILES."""
    
    # Load the trained model
    model = load_model(model_save_path)
    # Feature engineering (Fingerprint generation)
    X_fp, imgs = get_morgan_fingerprint(smile)

    if imgs[0] is None:
        raise ValueError("No image generated for the given SMILES. Please check your input.")

    # Predicting
    prediction = model.predict(X_fp)[0]
    probability = np.round((model.predict_proba(X_fp)[:, 1] * 100), 2)[0]
    print("Prediction made successfully...")

    img = imgs[0]
    buff = BytesIO()
    img.save(buff, format="PNG")
    img_b64 = base64.b64encode(buff.getvalue()).decode("utf-8")

    # Outputting the results
    results = pd.DataFrame({
        "SMILES": smile,
        "Molecule": img_b64,
        "Prediction": int(prediction),
        "Probability (%)": float(probability)
    })

    return results