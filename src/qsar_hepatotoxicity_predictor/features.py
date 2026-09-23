import pandas as pd
from rdkit.Chem import AllChem as Chem
from rdkit.Chem import Descriptors as Descs
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline

def descriptor_generator(df: pd.DataFrame, smiles_col: str = "SMILES") -> pd.DataFrame:
    """Generates the molecular descriptors from SMILES."""
    desc_names = [desc[0] for desc in Descs.descList]
    valid_indices = []
    desc_list = []

    for idx, smile in zip(df.index, df[smiles_col]): 
        mol = Chem.MolFromSmiles(smiles_col)
        if mol is None:
            return None

        # Sanitizes and adds hydrogens to the molecules
        Chem.SanitizeMol(mol, catchErrors=True)
        Chem.AddHs(mol)

        # Calculates the molecular descriptors
        desc_dict = Descs.CalcMolDescriptors(mol)
        desc_list.append(desc_dict)
        valid_indices.append(idx)

    # Converts descriptors to a DataFrame and combines with the original DataFrame
    desc_df = pd.DataFrame(desc_list, index=valid_indices)
    combined_df = pd.concat([df.loc[valid_indices], desc_df], axis=1)

    print(f"Descriptor generation completed: {len(desc_names)} descritpors for {len(combined_df)} molecules")
    return combined_df                        
