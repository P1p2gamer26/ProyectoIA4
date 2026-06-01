#!/usr/bin/env python3
"""
Ejecuta el experimento factorial completo (24 runs), genera tabla, análisis,
mejores modelos y matrices de confusión.

Uso (desde la raíz del proyecto):
    python scripts/run_experiment.py
"""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
os.chdir(ROOT)

import pandas as pd
from itertools import product

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedShuffleSplit

from preprocessing import encode_and_split
from paths import ensure_results_dirs
from plots import plot_experiment_bars, plot_confusion_matrix, plot_confusion_matrices_combined, plot_metrics_bar
from evaluation import evaluate
from experiment_analysis import generate_comparative_analysis
from best_models import select_best_configs, save_best_configs, train_best_models, SVM_SAMPLE

PROCESSED_PATH = "data/processed/datos_limpios.csv"
RESULTS_PATH = "results/"


def run_rf_experiment(X_train, y_train, X_test, y_test):
    rf_params = {
        "n_estimators": [100, 200],
        "max_depth": [10, 20],
        "min_samples_split": [5, 10],
    }
    results = []
    for n_est, depth, split in product(*rf_params.values()):
        model = RandomForestClassifier(
            n_estimators=n_est,
            max_depth=depth,
            min_samples_split=split,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results.append({
            "Técnica": "RF",
            "n_estimators": n_est,
            "max_depth": depth,
            "min_samples_split": split,
            "Accuracy": round(accuracy_score(y_test, y_pred), 4),
            "F1 (weighted)": round(f1_score(y_test, y_pred, average="weighted"), 4),
        })
        print(f"  RF | n={n_est} d={depth} s={split} → Acc={results[-1]['Accuracy']}")
    return pd.DataFrame(results)


def run_svm_experiment(X_train_sc, y_train, X_test_sc, y_test):
    svm_params = {"C": [1, 10], "kernel": ["rbf", "poly"], "gamma": ["scale", "auto"]}
    sss = StratifiedShuffleSplit(n_splits=1, train_size=SVM_SAMPLE, random_state=42)
    idx, _ = next(sss.split(X_train_sc, y_train))
    X_tr, y_tr = X_train_sc[idx], y_train.iloc[idx]
    results = []
    for c, kernel, gamma in product(*svm_params.values()):
        model = SVC(C=c, kernel=kernel, gamma=gamma, class_weight="balanced", random_state=42)
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_test_sc)
        results.append({
            "Técnica": "SVM",
            "C": c,
            "kernel": kernel,
            "gamma": gamma,
            "Accuracy": round(accuracy_score(y_test, y_pred), 4),
            "F1 (weighted)": round(f1_score(y_test, y_pred, average="weighted"), 4),
        })
        print(f"  SVM | C={c} k={kernel} g={gamma} → Acc={results[-1]['Accuracy']}")
    return pd.DataFrame(results)


def run_mlp_experiment(X_train_sc, y_train, X_test_sc, y_test):
    mlp_params = {
        "hidden_layer_sizes": [(100, 50), (200, 100)],
        "activation": ["relu", "tanh"],
        "learning_rate_init": [0.001, 0.01],
    }
    results = []
    for layers, act, lr in product(*mlp_params.values()):
        model = MLPClassifier(
            hidden_layer_sizes=layers,
            activation=act,
            learning_rate_init=lr,
            max_iter=200,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            verbose=False,
        )
        model.fit(X_train_sc, y_train)
        y_pred = model.predict(X_test_sc)
        results.append({
            "Técnica": "MLP",
            "hidden_layers": str(layers),
            "activation": act,
            "learning_rate": lr,
            "Accuracy": round(accuracy_score(y_test, y_pred), 4),
            "F1 (weighted)": round(f1_score(y_test, y_pred, average="weighted"), 4),
        })
        print(f"  MLP | {layers} {act} lr={lr} → Acc={results[-1]['Accuracy']}")
    return pd.DataFrame(results)


def main():
    ensure_results_dirs(RESULTS_PATH.rstrip("/"))
    print("Cargando datos...")
    df = pd.read_csv(PROCESSED_PATH)
    X_train, X_test, X_train_sc, X_test_sc, y_train, y_test, le = encode_and_split(df)
    print(f"Train: {X_train.shape[0]:,} | Test: {X_test.shape[0]:,}")

    print("\n=== Experimento RF ===")
    df_rf = run_rf_experiment(X_train, y_train, X_test, y_test)

    print("\n=== Experimento SVM ===")
    df_svm = run_svm_experiment(X_train_sc, y_train, X_test_sc, y_test)

    print("\n=== Experimento MLP ===")
    df_mlp = run_mlp_experiment(X_train_sc, y_train, X_test_sc, y_test)

    df_exp = pd.concat([
        df_rf.rename(columns={
            "n_estimators": "param1", "max_depth": "param2", "min_samples_split": "param3"
        })[["Técnica", "param1", "param2", "param3", "Accuracy", "F1 (weighted)"]],
        df_svm.rename(columns={"C": "param1", "kernel": "param2", "gamma": "param3"})[
            ["Técnica", "param1", "param2", "param3", "Accuracy", "F1 (weighted)"]
        ],
        df_mlp.rename(columns={
            "hidden_layers": "param1", "activation": "param2", "learning_rate": "param3"
        })[["Técnica", "param1", "param2", "param3", "Accuracy", "F1 (weighted)"]],
    ], ignore_index=True)
    table_path = RESULTS_PATH + "experiment_table.csv"
    df_exp.to_csv(table_path, index=False)
    print(f"\n✅ Tabla: {table_path}")

    plot_experiment_bars(df_rf, df_svm, df_mlp, RESULTS_PATH + "experimento_accuracy.png")

    analysis = generate_comparative_analysis(
        df_rf, df_svm, df_mlp, SVM_SAMPLE, len(X_train)
    )
    analysis_path = RESULTS_PATH + "analisis_comparativo.md"
    with open(analysis_path, "w", encoding="utf-8") as f:
        f.write(analysis)
    print(f"✅ Análisis: {analysis_path}")
    print("\n" + analysis)

    configs = select_best_configs(df_rf, df_svm, df_mlp)
    save_best_configs(configs, RESULTS_PATH + "best_configs.json")
    print(f"\n✅ Mejores configs: {RESULTS_PATH}best_configs.json")

    print("\n=== Reentrenando mejores modelos ===")
    trained = train_best_models(
        configs, X_train, X_test, X_train_sc, X_test_sc, y_train, y_test
    )
    CM_PATH = RESULTS_PATH + "confusion_matrices/"
    preds = {}
    all_metrics = {}
    for name, (model, y_pred) in trained.items():
        X_eval = X_test if name == "Random Forest" else X_test_sc
        _, metrics, _ = evaluate(model, X_eval, y_test, le)
        all_metrics[name] = metrics
        preds[name] = y_pred
        fname = name.lower().replace(" ", "_")
        plot_confusion_matrix(
            y_test, y_pred, le.classes_,
            f"Matriz de Confusión — {name} (mejor config.)",
            CM_PATH + f"cm_{fname}.png",
        )

    plot_confusion_matrices_combined(
        preds, y_test, le.classes_, CM_PATH + "cm_combinadas_mejores.png"
    )
    plot_metrics_bar(all_metrics, RESULTS_PATH + "comparacion_modelos_experimento.png")
    print(f"✅ Matrices en {CM_PATH}")
    print("✅ Experimento completado.")


if __name__ == "__main__":
    main()
