#!/usr/bin/env python3
"""Gera um dia de dados simulados para substituir leituras do SCT-013-30.

São produzidas 43.200 leituras, uma a cada 2 segundos. O arquivo contém apenas
as colunas que o sensor forneceria: Time, Unix e Aggregate.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def gerar_dados(inicio, seed=42):
    rng = np.random.default_rng(seed)
    periodos = 24 * 60 * 60 // 2
    timestamps = pd.date_range(start=inicio, periods=periodos, freq="2s")
    segundos = np.arange(periodos) * 2
    hora = (segundos / 3600.0) % 24

    # Carga de fundo residencial: varia lentamente ao longo do dia.
    base = 65 + 18 * np.sin(2 * np.pi * (hora - 6) / 24)
    base += rng.normal(0, 4, periodos)
    base = np.clip(base, 35, None)

    # Geladeira: ciclos intermitentes de aproximadamente 85 W.
    ciclo_fridge = ((segundos + 900) % 2400) < 1150
    appliance1 = np.where(ciclo_fridge, 85 + rng.normal(0, 4, periodos), 0)
    appliance1 = np.clip(appliance1, 0, None)

    # Máquina de lavar: dois possíveis ciclos, com picos de aquecimento.
    ciclo_manha = (hora >= 8.0) & (hora < 9.2)
    ciclo_noite = (hora >= 18.0) & (hora < 19.3)
    ciclo = ciclo_manha | ciclo_noite
    fase = (segundos % 180) / 180
    potencia_maquina = np.select(
        [fase < 0.25, fase < 0.55, fase < 0.80],
        [250, 1400, 500],
        default=80,
    )
    appliance5 = np.where(ciclo, potencia_maquina, 0.0)
    appliance5 += np.where(ciclo, rng.normal(0, 25, periodos), 0)
    appliance5 = np.clip(appliance5, 0, None)

    # TV: uso concentrado no almoço e à noite.
    tv_ligada = ((hora >= 12) & (hora < 14)) | ((hora >= 19) & (hora < 23))
    appliance8 = np.where(tv_ligada, 65 + rng.normal(0, 3, periodos), 0)
    appliance8 = np.clip(appliance8, 0, None)

    # Ruído de medição do SCT-013-30 e arredondamento do equipamento IoT.
    agregado = base + appliance1 + appliance5 + appliance8
    agregado += rng.normal(0, 5, periodos)
    agregado = np.clip(agregado, 0, None)

    # Somente estas três colunas são enviadas pelo sensor/modelo de aquisição.
    # As cargas individuais devem ser geradas posteriormente pelo modelo NILM.
    return pd.DataFrame({
        "Time": timestamps.strftime("%Y-%m-%d %H:%M:%S"),
        "Unix": timestamps.astype("int64") // 10**9,
        "Aggregate": np.rint(agregado).astype("int32"),
    })


def main():
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(base / "sct013_30_simulado_1dia.csv"))
    parser.add_argument("--start", default="2026-10-27 00:00:00")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    data = gerar_dados(args.start, args.seed)
    data.to_csv(output, index=False)
    print(f"Arquivo criado: {output}")
    print(f"Leituras: {len(data):,} | intervalo: 2 segundos | duração: 24 horas")


if __name__ == "__main__":
    main()
