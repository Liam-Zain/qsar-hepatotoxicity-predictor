import os
import pandas as pd

def load_tox_khan(file_path: str) -> pd.DataFrame:
    """Loads the Tox Khan dataset from a CSV file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    df = pd.read_csv(file_path)
    print(f"Tox Khan dataset loaded successfully: {len(df)} molecules")
    return df