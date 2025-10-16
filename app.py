import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import requests
from datetime import timedelta
from multiprocessing import Process
import uvicorn
import time

# -------------------------
# Função que roda o Uvicorn
# -------------------------
def start_uvicorn():
    uvicorn.run("src.api.main:app", host="127.0.0.1", port=8000, reload=False)

def is_api_ready():
    try:
        response = requests.get("http://127.0.0.1:8000/health", timeout=2)
        return response.status_code == 200
    except:
        return False

def plot_line_chart(df, prediction=None):
    fig = go.Figure()
    # Linha principal
    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['Close'],
        mode='lines',
        name='Close Price',
        line=dict(color='#2E86AB', width=3),
        fill='tozeroy',
        fillcolor='rgba(46,134,171,0.1)'
    ))
    # Previsão
    if prediction is not None:
        last_date = df['Date'].iloc[-1]
        pred_date = last_date + timedelta(days=1)
        fig.add_trace(go.Scatter(
            x=[pred_date],
            y=[prediction],
            mode='markers+text',
            name='Predicted Close',
            marker=dict(size=14, color='#E74C3C', symbol='x'),
            text=[f"{prediction:.2f}"],
            textposition="top center",
            textfont=dict(size=14, color='#E74C3C')
        ))
    fig.update_layout(
        title=dict(text='📊 Preços de Fechamento (Close)', x=0.5, font=dict(size=22)),
        xaxis_title='Data',
        yaxis_title='Preço (R$)',
        template='plotly_white',
        hovermode='x unified',
        height=600,
        font=dict(family="Arial", size=12),
        legend=dict(title='Legenda', font=dict(size=12))
    )
    return fig

def main():
    # -------------------------
    # CSS customizado
    # -------------------------
    st.markdown("""
        <style>
        .stApp { background-color: #F7F9FB; color:#333333; }
        .block-container {
            padding: 2rem;
            border-radius: 12px;
            box-shadow: 0 6px 10px rgba(0,0,0,0.08);
            background-color: #ffffff;
        }
        .stButton > button {
            background-color: #2E86AB;
            color: white;
            border-radius: 6px;
            padding: 0.6rem 1.2rem;
            font-size: 16px;
            font-weight: bold;
        }
        .stButton > button:hover { background-color: #1B4F72; }
        h1, h2, h3, h4 { color: #1B2631; }
        .sidebar .sidebar-content {
            background-color: #D6EAF8;
            color: #1B2631;
        }
        </style>
    """, unsafe_allow_html=True)

    st.title("📈 LSTM Stock Prediction Dashboard")

    # -------------------------
    # Start API
    # -------------------------
    if 'uvicorn_process' not in st.session_state:
        st.session_state.uvicorn_process = None

    if st.session_state.uvicorn_process is None:
        with st.spinner("Iniciando API..."):
            p = Process(target=start_uvicorn)
            p.start()
            st.session_state.uvicorn_process = p
            start_time = time.time()
            while not is_api_ready():
                if time.time() - start_time > 30:
                    st.error("Falha ao iniciar API")
                    break
                time.sleep(1)
            else:
                st.success("API pronta em http://127.0.0.1:8000")

    # -------------------------
    # Load df
    # -------------------------
    df = pd.read_csv("data/processed/df.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    st.subheader("Gráfico de Preços de Fechamento")
    chart = plot_line_chart(df)
    st.plotly_chart(chart, use_container_width=True)

    # -------------------------
    # Botão Predict
    # -------------------------
    if st.button("Predict"):
        if not is_api_ready():
            st.error("API não está disponível")
        else:
            with st.spinner("Executando previsão..."):
                df_prev = pd.read_csv("data/processed/df_prev.csv")
                df_prev['Date_numeric'] = range(len(df_prev))
                sequence = df_prev.drop(columns=['Close']).tail(30)[
                    ['Date_numeric','Open','High','Low','Volume']
                ].values.tolist()
                payload = {"sequence": sequence}

                try:
                    response = requests.post("http://127.0.0.1:8000/predict", json=payload, timeout=10)
                    if response.status_code == 200:
                        prediction = response.json()['prediction']
                        st.success(f"Previsão obtida: {prediction:.2f}")
                        updated_chart = plot_line_chart(df, prediction=prediction)
                        st.plotly_chart(updated_chart, use_container_width=True)
                    else:
                        st.error(f"Erro na API: {response.status_code} - {response.text}")
                except Exception as e:
                    st.error(f"Erro durante previsão: {e}")

if __name__ == "__main__":
    main()
