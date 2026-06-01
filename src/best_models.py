"""
best_models.py
Entrenamiento de los mejores modelos (según experimento) y exportación de configs.
"""
import json

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedShuffleSplit

from experiment_analysis import parse_mlp_layers

SVM_SAMPLE = 15_000


def select_best_configs(df_rf, df_svm, df_mlp) -> dict:
    """Selecciona la mejor fila por técnica usando F1 ponderado."""
    best_rf = df_rf.loc[df_rf["F1 (weighted)"].idxmax()]
    best_svm = df_svm.loc[df_svm["F1 (weighted)"].idxmax()]
    best_mlp = df_mlp.loc[df_mlp["F1 (weighted)"].idxmax()]
    return {
        "RF": {
            "n_estimators": int(best_rf["n_estimators"]),
            "max_depth": int(best_rf["max_depth"]),
            "min_samples_split": int(best_rf["min_samples_split"]),
            "accuracy": float(best_rf["Accuracy"]),
            "f1": float(best_rf["F1 (weighted)"]),
        },
        "SVM": {
            "C": float(best_svm["C"]),
            "kernel": str(best_svm["kernel"]),
            "gamma": str(best_svm["gamma"]),
            "accuracy": float(best_svm["Accuracy"]),
            "f1": float(best_svm["F1 (weighted)"]),
        },
        "MLP": {
            "hidden_layer_sizes": parse_mlp_layers(str(best_mlp["hidden_layers"])),
            "activation": str(best_mlp["activation"]),
            "learning_rate_init": float(best_mlp["learning_rate"]),
            "accuracy": float(best_mlp["Accuracy"]),
            "f1": float(best_mlp["F1 (weighted)"]),
        },
    }


def save_best_configs(configs: dict, path: str) -> None:
    """Guarda configuraciones en JSON (tuplas → listas)."""
    serializable = {}
    for tech, cfg in configs.items():
        serializable[tech] = dict(cfg)
        if "hidden_layer_sizes" in serializable[tech]:
            serializable[tech]["hidden_layer_sizes"] = list(
                serializable[tech]["hidden_layer_sizes"]
            )
    with open(path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)


def load_best_configs(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if "MLP" in data:
        data["MLP"]["hidden_layer_sizes"] = tuple(data["MLP"]["hidden_layer_sizes"])
    return data


def train_best_models(configs, X_train, X_test, X_train_sc, X_test_sc, y_train, y_test):
    """
    Entrena RF, SVM y MLP con las mejores hiperparámetros del experimento.

    Retorna dict nombre → (modelo, y_pred, metrics_dict)
    """
    results = {}

    rf_cfg = configs["RF"]
    rf = RandomForestClassifier(
        n_estimators=rf_cfg["n_estimators"],
        max_depth=rf_cfg["max_depth"],
        min_samples_split=rf_cfg["min_samples_split"],
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    results["Random Forest"] = (rf, y_pred_rf)

    svm_cfg = configs["SVM"]
    sss = StratifiedShuffleSplit(n_splits=1, train_size=SVM_SAMPLE, random_state=42)
    idx, _ = next(sss.split(X_train_sc, y_train))
    svm = SVC(
        C=svm_cfg["C"],
        kernel=svm_cfg["kernel"],
        gamma=svm_cfg["gamma"],
        class_weight="balanced",
        random_state=42,
    )
    svm.fit(X_train_sc[idx], y_train.iloc[idx])
    y_pred_svm = svm.predict(X_test_sc)
    results["SVM"] = (svm, y_pred_svm)

    mlp_cfg = configs["MLP"]
    mlp = MLPClassifier(
        hidden_layer_sizes=mlp_cfg["hidden_layer_sizes"],
        activation=mlp_cfg["activation"],
        learning_rate_init=mlp_cfg["learning_rate_init"],
        max_iter=200,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        verbose=False,
    )
    mlp.fit(X_train_sc, y_train)
    y_pred_mlp = mlp.predict(X_test_sc)
    results["MLP"] = (mlp, y_pred_mlp)

    return results
