from prometheus_client import Counter, Histogram
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from prometheus_client import make_asgi_app


REQUEST_COUNT = Counter('api_request_count', 'Total HTTP requests', ['method', 'endpoint', 'http_status'])
REQUEST_LATENCY = Histogram('api_request_latency_seconds', 'Request latency', ['endpoint'])


class PrometheusMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        method = request.method
        with REQUEST_LATENCY.labels(path).time():
            response = await call_next(request)
        REQUEST_COUNT.labels(method, path, str(response.status_code)).inc()
        return response


# ASGI app para /metrics
metrics_app = make_asgi_app()