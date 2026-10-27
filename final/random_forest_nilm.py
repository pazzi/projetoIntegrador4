#!/usr/bin/env python3
"""Baseline Random Forest para comparação com o modelo de deep learning.

Avalia Appliance1 (Geladeira), Appliance5 (Máquina de lavar roupas) e
Appliance8 (TV) usando o agregado e características temporais derivadas.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score


APPLIANCE_COLS = ["Appliance1", "Appliance5", "Appliance8"]
APPLIANCE_NAMES = ["Geladeira", "Máquina de lavar roupas", "TV"]


def load_features(path, max_rows=0, state_threshold=10.0):
    columns = ["Unix", "Aggregate", *APPLIANCE_COLS]
    df = pd.read_csv(path, usecols=columns, nrows=max_rows or None)
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    df = df.sort_values("Unix", kind="mergesort").reset_index(drop=True)

    aggregate = df["Aggregate"].astype("float32")
    delta = aggregate.diff().fillna(0.0)
    features = pd.DataFrame({
        "aggregate": aggregate,
        "delta": delta,
        "mean_5": aggregate.rolling(5, min_periods=1).mean(),
        "peak_5": aggregate.rolling(5, min_periods=1).max(),
        "mean_20": aggregate.rolling(20, min_periods=1).mean(),
        "peak_20": aggregate.rolling(20, min_periods=1).max(),
    })
    targets = (df[APPLIANCE_COLS].to_numpy() >= state_threshold).astype(np.int8)
    return features.to_numpy(np.float32), targets


def tune_threshold(probabilities, targets):
    best_thresholds = []
    for column in range(targets.shape[1]):
        best_f1, best = -1.0, 0.5
        for threshold in np.arange(0.05, 0.96, 0.01):
            predicted = probabilities[:, column] >= threshold
            score = f1_score(targets[:, column], predicted, zero_division=0)
            if score > best_f1:
                best_f1, best = score, float(threshold)
        best_thresholds.append(best)
    return np.asarray(best_thresholds, dtype=np.float32)


def sample_rows(features, targets, max_samples, rng, prioritize_rare=True):
    """Reduz o conjunto mantendo todas as linhas positivas disponíveis."""
    if max_samples <= 0 or len(features) <= max_samples:
        return features, targets
    if not prioritize_rare:
        selected = rng.choice(len(features), size=max_samples, replace=False)
        return features[selected], targets[selected]
    positive_rows = np.flatnonzero(targets.any(axis=1))
    # Prioriza os positivos da Appliance5, que é a carga mais rara.
    rare_rows = np.flatnonzero(targets[:, 1] == 1)
    if len(rare_rows) >= max_samples:
        selected = rng.choice(rare_rows, size=max_samples, replace=False)
    else:
        other_positive = np.setdiff1d(positive_rows, rare_rows, assume_unique=True)
        remaining = max_samples - len(rare_rows)
        negative_rows = np.flatnonzero(~targets.any(axis=1))
        candidates = np.concatenate([other_positive, negative_rows])
        chosen_other = rng.choice(candidates, size=min(remaining, len(candidates)), replace=False)
        selected = np.concatenate([rare_rows, chosen_other])
    rng.shuffle(selected)
    return features[selected], targets[selected]


def main():
    script_dir = Path(__file__).resolve().parent
    default_input = script_dir / "CLEAN_House1.csv"
    if not default_input.exists():
        default_input = script_dir / "model" / "CLEAN_House1.csv"
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(default_input))
    parser.add_argument("--output-dir", default=str(script_dir / "random_forest_results"))
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--max-train-samples", type=int, default=100000)
    parser.add_argument("--max-eval-samples", type=int, default=200000,
                        help="limite de amostras para validação e teste")
    parser.add_argument("--trees", type=int, default=100)
    parser.add_argument("--n-jobs", type=int, default=4,
                        help="número de núcleos usados pelo Random Forest")
    parser.add_argument("--state-threshold", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    input_path = Path(args.input).expanduser()
    if not input_path.is_absolute() and not input_path.exists():
        input_path = script_dir / input_path
    if not input_path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {input_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Carregando {input_path}...")
    x, y = load_features(str(input_path), args.max_rows, args.state_threshold)
    n = len(x)
    train_end, validation_end = int(n * 0.70), int(n * 0.85)
    x_train, y_train = x[:train_end], y[:train_end]
    x_val, y_val = x[train_end:validation_end], y[train_end:validation_end]
    x_test, y_test = x[validation_end:], y[validation_end:]

    rng = np.random.default_rng(args.seed)
    x_train, y_train = sample_rows(x_train, y_train, args.max_train_samples, rng)
    # Validação e teste mantêm a distribuição original para que o F1 seja
    # comparável ao TCN; apenas o treino recebe prioridade para Appliance5.
    x_val, y_val = sample_rows(x_val, y_val, args.max_eval_samples, rng, False)
    x_test, y_test = sample_rows(x_test, y_test, args.max_eval_samples, rng, False)

    print(f"Linhas: {n:,} | treino usado: {len(x_train):,} | "
          f"validação: {len(x_val):,} | teste: {len(x_test):,}")

    probabilities_val = np.zeros((len(x_val), len(APPLIANCE_COLS)), dtype=np.float32)
    probabilities_test = np.zeros((len(x_test), len(APPLIANCE_COLS)), dtype=np.float32)
    models = []
    for column, name in enumerate(APPLIANCE_NAMES):
        model = RandomForestClassifier(
            n_estimators=args.trees,
            max_features="sqrt",
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=args.n_jobs,
            random_state=args.seed,
        )
        model.fit(x_train, y_train[:, column])
        probabilities_val[:, column] = model.predict_proba(x_val)[:, 1]
        probabilities_test[:, column] = model.predict_proba(x_test)[:, 1]
        models.append(model)
        print(f"Treinado: {name}")

    thresholds = tune_threshold(probabilities_val, y_val)
    predicted_test = probabilities_test >= thresholds
    results = {}
    for column, name in enumerate(APPLIANCE_NAMES):
        results[name] = {
            "f1": float(f1_score(y_test[:, column], predicted_test[:, column], zero_division=0)),
            "precision": float(precision_score(y_test[:, column], predicted_test[:, column], zero_division=0)),
            "recall": float(recall_score(y_test[:, column], predicted_test[:, column], zero_division=0)),
            "accuracy": float(accuracy_score(y_test[:, column], predicted_test[:, column])),
            "threshold": float(thresholds[column]),
        }

    result_df = pd.DataFrame(results).T
    print("\nMétricas Random Forest no teste:")
    print(result_df.round(4).to_string())
    result_df.to_csv(output_dir / "metricas_random_forest.csv")
    with open(output_dir / "metricas_random_forest.json", "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False)
    joblib.dump(models, output_dir / "random_forest_models.joblib")
    print(f"\nResultados salvos em {output_dir}")


if __name__ == "__main__":
    main()
