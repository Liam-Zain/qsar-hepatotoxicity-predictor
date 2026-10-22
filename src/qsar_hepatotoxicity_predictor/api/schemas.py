from pydantic import BaseModel

class SMILESRequest(BaseModel):
    smiles: str

class BatchSMILESRequest(BaseModel):
    smiles_list: list[str]

class PredictionResponse(BaseModel):
    smiles: str
    prediction: int
    probability: float
    is_hepatotoxic: bool
    likelihood: str