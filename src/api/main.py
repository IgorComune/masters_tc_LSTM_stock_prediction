from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from src.api.schemas import PredictRequest, PredictResponse
from src.api.predictor import Predictor
from src.api.monitoring import PrometheusMiddleware, metrics_app
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('api')
app = FastAPI(title='LSTM Prediction API')

# CORS básico
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
    )

# Prometheus
app.add_middleware(PrometheusMiddleware)
app.mount('/metrics', metrics_app)
predictor = Predictor()
@app.on_event('startup')
async def startup_load_model():
    try:
        predictor.load()
        logger.info('Modelo carregado com sucesso no startup')
    except Exception as e:
        logger.warning(f'Falha ao carregar modelo no startup: {e}')

@app.get('/health')
async def health():
    return {'status': 'ok'}

@app.post('/predict', response_model=PredictResponse)
async def predict(req: PredictRequest):
    try:
        res = predictor.predict(req.sequence)
        return PredictResponse(probabilities=res['probabilities'],
            predicted_class=res['predicted_class'], details={})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    # endpoint para recarregar modelo sem reiniciar (útil para desenvolvimento)

@app.post('/reload')
async def reload_model(background_tasks: BackgroundTasks):
    def _reload():
        predictor.load()
        background_tasks.add_task(_reload)
    return {'reloading': True}