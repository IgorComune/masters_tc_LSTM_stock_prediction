FROM python:3.12-slim
WORKDIR /app
RUN pip3 install yfinance
RUN pip3 install pandas
RUN pip3 install -U scikit-learn
RUN pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130
RUN pip3 install mlflow
RUN pip3 install prometheus-client
RUN pip3 install psutil
COPY . .
EXPOSE 8000
HEALTHCHECK CMD curl --fail http://localhost:8000/health || exit 1
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]