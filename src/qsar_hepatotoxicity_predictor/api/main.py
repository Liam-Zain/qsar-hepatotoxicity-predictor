from fastapi import FastAPI

from qsar_hepatotoxicity_predictor.api.routes import router

app = FastAPI(
    title="QSAR Hepatotoxicity Predictor",
    description="API for predicting the potential hepatotoxicity of molecules using SMILES string",
    version="1.0.0"
)

app.include_router(router)