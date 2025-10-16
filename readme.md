Python 3.10.11
python -m venv .venv
.venv\Scripts\activate
python.exe -m pip install  --upgrade
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install numpy pandas matplotlib seaborn scikit-learn yfinance fastapi uvicorn pydantic python-multipart streamlit plotly tensorboard torch-summary tqdm joblib python-dotenv requests pytest

tickers list
https://www.dadosdemercado.com.br/acoes



uvicorn src.api.main:app --reload --port 8000
