import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

def get_morgan_fingerprint(smiles: np.ndarray, radius: int = 2, n_bits: int = 2048):
    """Generates the Morgan Fingerprints (ECFP4) from SMILES."""
    fingerprints = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            fp = rdFingerprintGenerator.GetMorganGenerator(mol, radius=radius, nBits=n_bits, includeChirality=True)
            fingerprints.append(np.array(fp))

        else:
            fingerprints.append(np.zeros(n_bits))
    
    return np.array(fingerprints)
