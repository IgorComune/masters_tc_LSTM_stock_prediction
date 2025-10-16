from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from src.api.schemas import PredictRequest, PredictResponse
from src.api.predictor import Predictor
from src.api.monitoring import PrometheusMiddleware, metrics_app
import logging
from pathlib import Path
import torch
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('api')
app = FastAPI(title='LSTM Prediction API')

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

# Prometheus metrics
app.add_middleware(PrometheusMiddleware)
app.mount('/metrics', metrics_app)

# Determine input_size from CSV, excluding only 'Close'
csv_path = "data/processed/df.csv"
df = pd.read_csv(csv_path)
input_size = df.drop(columns=['Close']).shape[1]  # Include 'Date' (5 features)

# Initialize Predictor with correct parameters
predictor = Predictor(
    model_path=r"models/final_model.pth",
    scaler_dir=r"models/scalers",
    window_size=30,  # Matches training
    input_size=input_size,  # Matches 5 features
    device="cuda" if torch.cuda.is_available() else "cpu"
)

@app.on_event('startup')
async def startup_load_model():
    try:
        predictor.load()
        logger.info('Modelo carregado com sucesso no startup')
    except Exception as e:
        logger.error(f'Falha ao carregar modelo no startup: {e}')
        raise

@app.get('/health')
async def health():
    return {'status': 'ok'}

@app.post('/predict', response_model=PredictResponse)
async def predict(req: PredictRequest):
    try:
        res = predictor.predict(seq=req.sequence, return_series=False)
        return PredictResponse(
            prediction=res['prediction'],
            details={}
        )
    except Exception as e:
        logger.error(f'Erro na predição: {e}')
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/reload')
async def reload_model(background_tasks: BackgroundTasks):
    def _reload():
        try:
            predictor.load()
            logger.info('Modelo recarregado com sucesso')
        except Exception as e:
            logger.error(f'Falha ao recarregar modelo: {e}')

    background_tasks.add_task(_reload)
    return {'reloading': True}