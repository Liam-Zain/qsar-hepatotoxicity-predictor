import pandas as pd

def load_dataset(file_path: str) -> pd.DataFrame:
    """Loads the raw Tox Khan dataset from a CSV file."""
    df = pd.read_csv(file_path)
    print(f"Tox Khan dataset loaded successfully: {len(df)} molecules")
    return df

def split_data(df: pd.DataFrame):
    """Splits the dataframe into features (SMILES) and Target."""
    X = df["SMILES"].values
    y = df["Target"].values
    return X, y