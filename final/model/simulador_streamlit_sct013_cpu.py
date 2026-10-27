import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="SCT-013-30 | NILM CPU", page_icon="⚡", layout="wide")

APPLIANCES = {
    "Appliance1": {"nome": "Geladeira", "potencia_w": 85.0},
    "Appliance5": {"nome": "Máquina de lavar roupas", "potencia_w": 1350.0},
    "Appliance8": {"nome": "TV", "potencia_w": 65.0},
}
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA = BASE_DIR / "sct013_30_simulado_1dia.csv"
DEFAULT_MODELS = BASE_DIR.parent / "resultados_cpu" / "random_forest_models.joblib"
DEFAULT_THRESHOLDS = BASE_DIR.parent / "resultados_cpu" / "metricas_random_forest.json"


@st.cache_data(show_spinner="Carregando leituras do SCT-013-30...")
def carregar_dados(caminho):
    df = pd.read_csv(caminho, usecols=["Time", "Unix", "Aggregate"])
    df = df.sort_values("Unix", kind="mergesort").reset_index(drop=True)
    df["Time"] = pd.to_datetime(df["Time"])
    agregado = df["Aggregate"].astype(float)
    features = pd.DataFrame({
        "aggregate": agregado,
        "delta": agregado.diff().fillna(0.0),
        "mean_5": agregado.rolling(5, min_periods=1).mean(),
        "peak_5": agregado.rolling(5, min_periods=1).max(),
        "mean_20": agregado.rolling(20, min_periods=1).mean(),
        "peak_20": agregado.rolling(20, min_periods=1).max(),
    })
    return df, features.to_numpy(np.float32)


@st.cache_resource(show_spinner="Carregando modelos Random Forest...")
def carregar_modelos(caminho):
    return joblib.load(caminho)


@st.cache_data
def carregar_limiares(caminho):
    if not Path(caminho).exists():
        return {coluna: 0.5 for coluna in APPLIANCES}
    with open(caminho, encoding="utf-8") as arquivo:
        metricas = json.load(arquivo)
    return {coluna: metricas.get(coluna, {}).get("threshold", 0.5)
            for coluna in APPLIANCES}


st.title("⚡ Monitoramento NILM — SCT-013-30")
st.caption("Reprodução de leituras a cada 2 segundos usando Random Forest em CPU")

with st.sidebar:
    st.header("Configuração")
    tarifa = st.number_input("Tarifa (R$/kWh)", min_value=0.0, value=0.85, step=0.05)
    executar = st.checkbox("Executar reprodução automática", value=False)
    reiniciar = st.button("Reiniciar reprodução")

caminho_dados = DEFAULT_DATA
caminho_modelos = DEFAULT_MODELS
caminho_limiares = DEFAULT_THRESHOLDS

if reiniciar:
    st.session_state.indice_sct = 0
    st.session_state.consumo_sct = {coluna: 0.0 for coluna in APPLIANCES}
    st.session_state.ultimo_indice_acumulado_sct = -1

if not Path(caminho_dados).exists():
    st.error(f"Arquivo de leituras não encontrado: {caminho_dados}")
    st.stop()
if not Path(caminho_modelos).exists():
    st.error(f"Arquivo de modelos não encontrado: {caminho_modelos}")
    st.info("Execute primeiro o treinamento Random Forest para gerar o arquivo .joblib.")
    st.stop()

df, features = carregar_dados(caminho_dados)
modelos = carregar_modelos(caminho_modelos)
limiares = carregar_limiares(caminho_limiares)

if "indice_sct" not in st.session_state:
    st.session_state.indice_sct = 0
if "consumo_sct" not in st.session_state:
    st.session_state.consumo_sct = {coluna: 0.0 for coluna in APPLIANCES}
if "ultimo_indice_acumulado_sct" not in st.session_state:
    st.session_state.ultimo_indice_acumulado_sct = -1

indice = min(st.session_state.indice_sct, len(df) - 1)
leitura = df.iloc[indice]
caracteristicas = features[indice].reshape(1, -1)
intervalo = 2.0 if indice == 0 else max(1.0, float(leitura["Unix"] - df.iloc[indice - 1]["Unix"]))

probabilidades, estados = {}, {}
for coluna in APPLIANCES:
    modelo = modelos[coluna] if isinstance(modelos, dict) else modelos[list(APPLIANCES).index(coluna)]
    probabilidades[coluna] = float(modelo.predict_proba(caracteristicas)[0, 1])
    estados[coluna] = probabilidades[coluna] >= limiares[coluna]

if indice > st.session_state.ultimo_indice_acumulado_sct:
    for coluna, configuracao in APPLIANCES.items():
        if estados[coluna]:
            st.session_state.consumo_sct[coluna] += configuracao["potencia_w"] * intervalo / 3600000.0
    st.session_state.ultimo_indice_acumulado_sct = indice

col1, col2, col3, col4 = st.columns(4)
col1.metric("Leitura", f"{indice + 1:,} / {len(df):,}")
col2.metric("Horário", leitura["Time"].strftime("%d/%m %H:%M:%S"))
col3.metric("Potência agregada", f"{leitura['Aggregate']:.0f} W")
col4.metric("Tarifa", f"R$ {tarifa:.2f}/kWh")

st.subheader("Estado e custo acumulado por equipamento")
linhas = []
for coluna, configuracao in APPLIANCES.items():
    consumo = st.session_state.consumo_sct[coluna]
    linhas.append({
        "Equipamento": configuracao["nome"],
        "Estado": "LIGADO" if estados[coluna] else "DESLIGADO",
        "Confiança": f"{probabilidades[coluna] * 100:.1f}%",
        "Potência estimada (W)": configuracao["potencia_w"] if estados[coluna] else 0.0,
        "Consumo acumulado (kWh)": consumo,
        "Valor consumido (R$)": consumo * tarifa,
    })
resumo = pd.DataFrame(linhas)
resumo["Consumo acumulado (kWh)"] = resumo["Consumo acumulado (kWh)"].round(4)
resumo["Valor consumido (R$)"] = resumo["Valor consumido (R$)"].round(2)
st.dataframe(resumo, hide_index=True, use_container_width=True)

total_kwh = sum(st.session_state.consumo_sct.values())
total_custo = total_kwh * tarifa
horas = max((float(leitura["Unix"]) - float(df.iloc[0]["Unix"]) + intervalo) / 3600.0, intervalo / 3600.0)
projecao_mensal = total_custo / horas * 24 * 30
m1, m2, m3 = st.columns(3)
m1.metric("Consumo acumulado total", f"{total_kwh:.4f} kWh")
m2.metric("Valor acumulado", f"R$ {total_custo:.2f}")
m3.metric("Projeção mensal", f"R$ {projecao_mensal:.2f}")

st.subheader("Sinal agregado")
inicio_grafico = max(0, indice - 300)
st.line_chart(df.iloc[inicio_grafico:indice + 1].set_index("Time")["Aggregate"])

if executar and indice < len(df) - 1:
    time.sleep(2)
    st.session_state.indice_sct += 1
    st.rerun()
