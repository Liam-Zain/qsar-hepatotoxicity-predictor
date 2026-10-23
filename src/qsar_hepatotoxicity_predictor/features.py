import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, Draw

def get_morgan_fingerprint(smiles_input, radius: int = 2, n_bits: int = 2048):
    """Generates the Morgan Fingerprints (ECFP4) from SMILES."""
    if isinstance(smiles_input, str):
        smiles_list = [smiles_input]
    else:
        smiles_list = list(smiles_input)
    
    fingerprints = []
    images = []

    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits, includeChirality=True)

    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        
        if mol:
            fp = generator.GetCountFingerprintAsNumPy(mol)
            fingerprints.append(fp)

            try:
                img = Draw.MolToImage(mol)
                images.append(img)
            except Exception:
                images.append(None)

        else:
            fingerprints.append(np.zeros(n_bits))
            images.append(None)
    
    return np.array(fingerprints), images