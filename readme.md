# 📈 LSTM Stock Price Prediction

[![Python 3.12](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-31011/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Stock price prediction system for B3 (Brazilian Stock Exchange) using LSTM (Long Short-Term Memory) neural networks with RESTful API for production inference.

---

## 🎯 About the Project

This project implements a Deep Learning model based on LSTM to predict future prices of stocks listed on B3 (Brazilian Stock Exchange). The complete system includes:

- 🧠 **LSTM Model** trained with historical data
- 🔄 **Automated pipeline** for data collection and preprocessing
- 🚀 **RESTful API** for real-time inference
- 📊 **Performance monitoring** and metrics
- 📈 **Evaluation** with MAE, RMSE, MAPE and R²

**Tech Challenge - Phase 4** | Post-Graduate in Machine Learning Engineering

---

## 📋 Table of Contents

- [Requirements](#-requirements)
- [Installation](#-installation)
- [Project Structure](#-project-structure)
- [Usage](#-usage)
- [API Endpoints](#-api-endpoints)
- [Model Architecture](#-model-architecture)
- [Performance Metrics](#-performance-metrics)
- [Notebooks](#-notebooks)
- [Deployment](#-deployment)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🔧 Requirements

### System
- **Python:** 3.12
- **GPU:** NVIDIA CUDA compatible (optional, but recommended)
- **Operating System:** Linux Ubuntu (recommended)
---

## 🚀 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/masters_tc_LSTM_stock_prediction.git
cd masters_tc_LSTM_stock_prediction
```

### 2. Create Virtual Environment
```bash
python -m venv .venv
```

### 3. Activate Virtual Environment

**Windows:**
```bash
.venv\Scripts\activate
```

**Linux/macOS:**
```bash
source .venv/bin/activate
```

### 4. Upgrade pip
```bash
python.exe -m pip install --upgrade pip
```

### 5. Install Dependencies
```bash
pip3 install yfinance
pip3 install pandas
pip3 install -U scikit-learn
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip3 install mlflow
pip3 install prometheus-client
```

---

## 📁 Project Structure

```
masters_tc_LSTM_stock_prediction/
│
├── data/                          # Project data
│   ├── raw/                       # Raw data
│   │   └── acoes-listadas-b3.csv  # B3 stock list
│   └── processed/                 # Processed data
│       └── df.csv                 # Preprocessed dataset
│
├── models/                        # Trained models
│   ├── checkpoints/               # Training checkpoints
│   │   ├── best_checkpoint.pth
│   │   └── best_model.pth
│   ├── scalers/                   # Scalers for normalization
│   │   ├── scaler_x.joblib
│   │   └── scaler_y.joblib
│   ├── final_model.pth            # Final production model
│   ├── final_model_metadata.json  # Model metadata
│   ├── training_history.json      # Training history
│   └── training_loss.png          # Training visualization
│
├── notebooks/                     # Jupyter Notebooks
│   ├── 01_data_collection.ipynb   # Data collection
│   └── 02_endpoints.ipynb         # API testing
│
├── src/                           # Source code
│   ├── api/                       # RESTful API
│   │   ├── main.py                # Main FastAPI app
│   │   ├── schemas.py             # Pydantic schemas
│   │   ├── predictor.py           # Inference logic
│   │   ├── monitoring.py          # Monitoring
│   │   └── test_api.py            # API tests
│   │
│   ├── models/                    # Model architecture
│   │   └── lstm_model.py          # LSTM implementation
│   │
│   └── utils/                     # Utilities
│       ├── data_collection.py     # Data collection
│       ├── model_training.py      # Training pipeline
│       └── model_evaluation.py    # Model evaluation
│
├── infer.py                       # CLI inference script
├── .gitignore                     # Ignored files
├── .gitattributes                 # Git configuration
├── LICENSE                        # MIT License
└── readme.md                      # This file
```

---

## 💻 Usage

### 1️⃣ Data Collection
**Data source:** [Yahoo Finance](https://finance.yahoo.com/) via `yfinance`
**B3 ticker list:** [Dados de Mercado](https://www.dadosdemercado.com.br/acoes)
---

### 2️⃣ Model Training

```python
# Train LSTM model
model, history = train_model(
    data_path="data/processed/df.csv",
    lookback=60,
    hidden_size=128,
    num_layers=2,
    dropout=0.2,
    epochs=150,
    batch_size=32,
    learning_rate=0.001
)
```

**Or via command line:**
```bash
python src/utils/model_training.py --epochs 150 --batch-size 32
```

---

### 3️⃣ Start the API

```bash
uvicorn src.api.main:app --reload --port 8000
```

API will be available at: **http://localhost:8000**

Interactive documentation: **http://localhost:8000/docs**

---

### 4️⃣ Make Predictions (CLI)

```bash
python infer.py
```

---

## 🌐 API Endpoints

### **GET** `/health`
Check API status

---

### **GET** `/model/info`
Returns model information

---

### **POST** `/predict`
Make future price prediction

---

### **POST** `/predict/multi`
Predict multiple future days

---

## 🧠 Model Architecture

### LSTM Network Structure

```
Input Layer (4 features)
    ↓
LSTM Layer 1 (128 hidden units)
    ↓
Dropout (0.2)
    ↓
LSTM Layer 2 (128 hidden units)
    ↓
Dropout (0.2)
    ↓
Fully Connected Layer (25 units, ReLU)
    ↓
Output Layer (1 unit)
```

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| **Lookback Window** | 60 days |
| **Hidden Size** | 128 |
| **Num Layers** | 2 |
| **Dropout** | 0.2 |
| **Learning Rate** | 0.001 |
| **Optimizer** | Adam |
| **Loss Function** | MSE |
| **Batch Size** | 32 |

### Input Features (X)
- Open
- High
- Low
- Volume

### Target (y)
- Close

---

## 📊 Performance Metrics

Results on test set:

| Metric | Value | Description |
|---------|-------|-----------|
| **MAE** | $0.18 | Mean Absolute Error |
| **RMSE** | $0.11 | Root Mean Squared Error |
| **MAPE** | 5.10% | Mean Absolute Percentage Error |
| **R²** | 0.98 | Coefficient of Determination |

**Interpretation:**
- ✅ **Low MAE:** Accurate predictions
- ✅ **MAPE < 5%:** Excellent for financial time series
- ✅ **R² > 0.85:** Model explains 87% of variance

---

## 📓 Notebooks

### 1. Data Collection (`01_data_collection.ipynb`)
- Historical data collection via `yfinance`
- Exploratory Data Analysis (EDA)
- Missing values treatment
- Outlier detection and correction
- Data normalization

### 2. API Testing (`02_endpoints.ipynb`)
- API endpoint testing
- Schema validation
- Usage examples
- Performance benchmarking

---

## 🐳 Deployment
###
```bash
uvicorn src.api.main:app --reload --port 8000
```
### Docker

```dockerfile
FROM python:3.12-slim
WORKDIR /app
RUN pip install yfinance
RUN pip install pandas
RUN pip3 install -U scikit-learn
RUN pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130
RUN pip install mlflow
COPY . .
EXPOSE 8000
HEALTHCHECK CMD curl --fail http://localhost:8000/health || exit 1
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Build and Run
```bash
docker build -t lstm-stock-api .
docker run -p 8000:8000 lstm-stock-api
```
---

## 🔍 Monitoring

Access: **http://localhost:8000/monitoring/metrics**

---

## 👥 Authors

**Tech Challenge - Phase 4**
- Igor Comune
- Éder Ray
- Mário Gotardelo
- Felippe Maurício

Post-Graduate in Machine Learning Engineering - FIAP

---

## 📄 License

This project is under the MIT license. See the [LICENSE](LICENSE) file for more details.

---

## 🙏 Acknowledgments

- [PyTorch](https://pytorch.org/) - Deep Learning Framework
- [FastAPI](https://fastapi.tiangolo.com/) - Modern web framework
- [Yahoo Finance](https://finance.yahoo.com/) - Data source
- [Dados de Mercado](https://www.dadosdemercado.com.br/) - B3 ticker list
- FIAP - Post-Graduate in Machine Learning Engineering

---

## ⚠️ Disclaimer

**This project is for educational purposes only.** It does not constitute financial advice. Always consult a qualified professional before making investment decisions. Past performance does not guarantee future results.