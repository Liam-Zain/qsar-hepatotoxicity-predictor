from pydantic import BaseModel
from typing import List

class SMILESRequest(BaseModel):
    smiles: str

class BatchSMILESRequest(BaseModel):
    smiles_list: List[str]

class PredictionResponse(BaseModel):
    smiles: str
    prediction: int
    probability: float
    is_hepatotoxic: bool
    likelihood: str