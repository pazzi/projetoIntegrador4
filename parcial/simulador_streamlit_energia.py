import streamlit as st
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier

# Configuração da página Streamlit
st.set_page_config(
    page_title="EnergIA - Dashboard IoT em Tempo Real",
    page_icon="⚡",
    layout="wide"
)

# Estilização CSS customizada
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stMetric {
        background-color: #ff2937;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #374151;
    }
    .status-on {
        color: #10b981;
        font-weight: bold;
    }
    .status-off {
        color: #ef4444;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Título e Cabeçalho
st.title("⚡ EnergIA: Monitoramento Residencial Não Invasivo (NILM)")
st.subheader("Simulação de Recepção de Dados IoT (ESP32 / SCT-013) & Desagregação por Machine Learning")

# Sidebar - Configurações da Simulação IoT
st.sidebar.header("🕹️ Controle do Dispositivo IoT")
frequencia_envio = st.sidebar.slider("Intervalo de Envio do ESP32 (segundos)", 1, 10, 2)
tarifa_kwh = st.sidebar.number_input("Tarifa de Energia (R$/kWh)", value=0.85, step=0.05)
limiar_tv = st.sidebar.slider("Limiar de Decisão da TV (Probability Threshold)", 0.1, 0.9, 0.30)

@st.cache_resource
def treinar_modelos_nilm():
    """Treina modelos Random Forest com a lógica do modelo_nilm_refit-v5.py."""
    np.random.seed(42)
    n_samples = 3000
    times = pd.date_range(start="2026-09-25", periods=n_samples, freq="8s")
    
    # Simulação de cargas
    fridge = np.random.choice([0, 90], size=n_samples, p=[0.55, 0.45]) + np.random.normal(0, 2, n_samples)
    fridge = np.maximum(0, fridge)
    
    washing = np.random.choice([0, 1400], size=n_samples, p=[0.88, 0.12])
    tv = np.random.choice([0, 70], size=n_samples, p=[0.65, 0.35]) + np.random.normal(0, 1, n_samples)
    tv = np.maximum(0, tv)
    base = np.random.uniform(40, 120, size=n_samples)
    
    aggregate = fridge + washing + tv + base
    
    df = pd.DataFrame({
        'Time': times,
        'Aggregate': aggregate,
        'Fridge': fridge,
        'WashingMachine': washing,
        'TV': tv
    })
    
    # Features
    df['Delta_Potencia'] = df['Aggregate'].diff().fillna(0)
    df['Potencia_Media_5m'] = df['Aggregate'].rolling(window=5, min_periods=1).mean()
    df['Potencia_Std_5m'] = df['Aggregate'].rolling(window=5, min_periods=1).std().fillna(0)
    df['Potencia_Std_3m'] = df['Aggregate'].rolling(window=3, min_periods=1).std().fillna(0)
    df['Hora'] = df['Time'].dt.hour
    df['Minuto'] = df['Time'].dt.minute
    df['Sin_Hora'] = np.sin(2 * np.pi * df['Hora'] / 24.0)
    df['Cos_Hora'] = np.cos(2 * np.pi * df['Hora'] / 24.0)
    
    # Targets
    df['Status_Fridge'] = (df['Fridge'] >= 15).astype(int)
    df['Status_Washing'] = (df['WashingMachine'] >= 20).astype(int)
    df['Status_TV'] = (df['TV'] >= 15).astype(int)
    
    features = ['Aggregate', 'Delta_Potencia', 'Potencia_Media_5m', 'Potencia_Std_5m', 'Potencia_Std_3m', 'Hora', 'Minuto', 'Sin_Hora', 'Cos_Hora']
    
    models = {}
    for app in ['Fridge', 'Washing', 'TV']:
        clf = RandomForestClassifier(n_estimators=50, max_depth=12, random_state=42, class_weight='balanced')
        clf.fit(df[features], df[f'Status_{app}'])
        models[app] = clf
        
    return models

modelos = treinar_modelos_nilm()

# Inicialização do histórico na sessão do Streamlit
if 'historico' not in st.session_state:
    st.session_state.historico = pd.DataFrame(columns=[
        'Timestamp', 'Potencia_Total', 'Prob_Geladeira', 'Prob_Maquina', 'Prob_TV',
        'Status_Geladeira', 'Status_Maquina', 'Status_TV'
    ])

# Botões de Controle da Streamlit App
col_ctrl1, col_ctrl2 = st.columns([1, 4])
with col_ctrl1:
    simular = st.checkbox("📡 Ativar Transmissão IoT", value=True)

# Geração de 1 dado simulado enviado pelo ESP32
def gerar_pacote_iot():
    agora = datetime.now()
    hora = agora.hour
    
    # Lógica de simulação de carga agregada com variação realista
    is_tv = 1 if (18 <= hora <= 23 or 11 <= hora <= 14) and np.random.rand() > 0.3 else 0
    is_geladeira = 1 if np.random.rand() > 0.5 else 0
    is_maquina = 1 if np.random.rand() > 0.9 else 0
    
    p_tv = (65 + np.random.normal(0, 3)) if is_tv else 0
    p_geladeira = (85 + np.random.normal(0, 5)) if is_geladeira else 0
    p_maquina = (1350 + np.random.normal(0, 20)) if is_maquina else 0
    p_fundo = np.random.uniform(50, 100)
    
    p_total = max(0, p_tv + p_geladeira + p_maquina + p_fundo)
    return agora, p_total

if simular:
    timestamp, p_total = gerar_pacote_iot()
    
    # Construção de atributos temporais e históricos para o modelo
    hist = st.session_state.historico
    ultimas_potencias = list(hist['Potencia_Total'].tail(4)) + [p_total]
    
    delta_p = p_total - ultimas_potencias[-2] if len(ultimas_potencias) > 1 else 0
    pot_media = np.mean(ultimas_potencias)
    pot_std = np.std(ultimas_potencias) if len(ultimas_potencias) > 1 else 0
    pot_std_3m = np.std(ultimas_potencias[-3:]) if len(ultimas_potencias) >= 3 else 0
    
    sin_h = np.sin(2 * np.pi * timestamp.hour / 24.0)
    cos_h = np.cos(2 * np.pi * timestamp.hour / 24.0)
    
    X_sample = pd.DataFrame([{
        'Aggregate': p_total,
        'Delta_Potencia': delta_p,
        'Potencia_Media_5m': pot_media,
        'Potencia_Std_5m': pot_std,
        'Potencia_Std_3m': pot_std_3m,
        'Hora': timestamp.hour,
        'Minuto': timestamp.minute,
        'Sin_Hora': sin_h,
        'Cos_Hora': cos_h
    }])
    
    # Inferência com os modelos carregados
    prob_g = modelos['Fridge'].predict_proba(X_sample)[0][1]
    prob_m = modelos['Washing'].predict_proba(X_sample)[0][1]
    prob_tv = modelos['TV'].predict_proba(X_sample)[0][1]
    
    stat_g = "LIGADO" if prob_g >= 0.32 else "DESLIGADO"
    stat_m = "LIGADO" if prob_m >= 0.10 else "DESLIGADO"
    stat_tv = "LIGADO" if prob_tv >= limiar_tv else "DESLIGADO"
    
    novo_registro = {
        'Timestamp': timestamp.strftime("%H:%M:%S"),
        'Potencia_Total': round(p_total, 1),
        'Prob_Geladeira': round(prob_g, 2),
        'Prob_Maquina': round(prob_m, 2),
        'Prob_TV': round(prob_tv, 2),
        'Status_Geladeira': stat_g,
        'Status_Maquina': stat_m,
        'Status_TV': stat_tv
    }
    
    st.session_state.historico = pd.concat([st.session_state.historico, pd.DataFrame([novo_registro])], ignore_index=True).tail(30)

# Layout de Métrica e Painéis
df_hist = st.session_state.historico

if not df_hist.empty:
    ultimo = df_hist.iloc[-1]
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("⚡ Potência Agregada (ESP32)", f"{ultimo['Potencia_Total']} W")
    
    col2.metric(
        "🧊 Geladeira", 
        ultimo['Status_Geladeira'], 
        f"Confiança: {int(ultimo['Prob_Geladeira']*100)}%"
    )
    col3.metric(
        "🧺 Máquina de Lavar", 
        ultimo['Status_Maquina'], 
        f"Confiança: {int(ultimo['Prob_Maquina']*100)}%"
    )
    col4.metric(
        "📺 Televisão", 
        ultimo['Status_TV'], 
        f"Confiança: {int(ultimo['Prob_TV']*100)}%"
    )

    st.markdown("---")
    
    # Gráficos em Tempo Real
    g_col1, g_col2 = st.columns([2, 1])
    
    with g_col1:
        st.subheader("📈 Série Temporal do Sinal Elétrico Agregado (Watts)")
        st.line_chart(df_hist.set_index('Timestamp')['Potencia_Total'])
        
    with g_col2:
        st.subheader("💰 Estimativa de Consumo & Custos")
        kw_medio = df_hist['Potencia_Total'].mean() / 1000.0
        custo_hora = kw_medio * tarifa_kwh
        custo_mes = custo_hora * 24 * 30
        
        st.write(f"**Consumo Médio Atual:** {kw_medio:.3f} kW")
        st.write(f"**Custo Estimado/Hora:** R$ {custo_hora:.2f}")
        st.metric("Projeção da Fatura Mensal", f"R$ {custo_mes:.2f}")
        
        # Alerta de desperdício
        if ultimo['Status_Maquina'] == "LIGADO" and ultimo['Potencia_Total'] > 1200:
            st.warning("⚠️ **Alerta EnergIA:** Elevado pico de consumo detectado (Máquina de lavar em ciclo de aquecimento/centrifugação).")

    # Tabela com o histórico das últimas leituras
    st.subheader("📋 Log de Telemetria e Inferência em Tempo Real")
    st.dataframe(df_hist, use_container_width=True)

# Atualização contínua se a simulação estiver ativa
if simular:
    time.sleep(frequencia_envio)
    st.rerun()
