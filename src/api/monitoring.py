from prometheus_client import Counter, Histogram, Gauge
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from prometheus_client import make_asgi_app
import psutil
import time

# ============= MÉTRICAS DE API (ORIGINAL) =============
REQUEST_COUNT = Counter('api_request_count', 'Total HTTP requests', ['method', 'endpoint', 'http_status'])
REQUEST_LATENCY = Histogram('api_request_latency_seconds', 'Request latency', ['endpoint'])

# ============= MÉTRICAS DO MODELO =============
MODEL_INFERENCE_TIME = Histogram(
    'model_inference_seconds',
    'Tempo de inferência do modelo',
    ['model_name']
)

PREDICTIONS_TOTAL = Counter(
    'predictions_total',
    'Total de predições realizadas',
    ['model_name', 'status']
)

PREDICTION_INPUT_SIZE = Histogram(
    'prediction_input_size_bytes',
    'Tamanho do input da predição',
    ['endpoint']
)

# ============= MÉTRICAS DE RECURSOS DO SISTEMA =============
CPU_USAGE = Gauge('system_cpu_usage_percent', 'Uso de CPU em percentual')
MEMORY_USAGE = Gauge('system_memory_usage_bytes', 'Uso de memória RAM em bytes')
MEMORY_PERCENT = Gauge('system_memory_usage_percent', 'Uso de memória RAM em percentual')
DISK_USAGE = Gauge('system_disk_usage_percent', 'Uso de disco em percentual')


class PrometheusMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        method = request.method
        
        # Atualizar métricas de recursos do sistema a cada requisição
        self.update_system_metrics()
        
        # Medir latência
        with REQUEST_LATENCY.labels(path).time():
            response = await call_next(request)
        
        # Contar requisições
        REQUEST_COUNT.labels(method, path, str(response.status_code)).inc()
        
        return response
    
    @staticmethod
    def update_system_metrics():
        """Atualiza métricas de utilização de recursos"""
        try:
            CPU_USAGE.set(psutil.cpu_percent(interval=0.1))
            memory = psutil.virtual_memory()
            MEMORY_USAGE.set(memory.used)
            MEMORY_PERCENT.set(memory.percent)
            disk = psutil.disk_usage('/')
            DISK_USAGE.set(disk.percent)
        except Exception as e:
            print(f"Erro ao coletar métricas do sistema: {e}")


# ============= FUNÇÃO PARA RASTREAR PREDIÇÕES =============
def track_prediction(model_name: str = "default"):
    """Decorator para rastrear tempo de inferência do modelo"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                PREDICTIONS_TOTAL.labels(model_name, "success").inc()
                return result
            except Exception as e:
                PREDICTIONS_TOTAL.labels(model_name, "error").inc()
                raise e
            finally:
                inference_time = time.time() - start_time
                MODEL_INFERENCE_TIME.labels(model_name).observe(inference_time)
        return wrapper
    return decorator


# ASGI app para /metrics
metrics_app = make_asgi_app()