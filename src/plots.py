"""
plots.py
Funciones de visualización para EDA, modelos y experimento.
Todas las funciones guardan la figura en save_path y la cierran.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay
from evaluation import get_confusion_matrix
from paths import ensure_parent_dir


def plot_confusion_matrix(y_test, y_pred, labels, title: str, save_path: str):
    """Heatmap de la matriz de confusión con anotaciones."""
    cm = get_confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Real")
    ax.set_title(title)
    plt.tight_layout()
    ensure_parent_dir(save_path)
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_confusion_matrices_combined(preds: dict, y_test, labels, save_path: str):
    """
    Tres matrices de confusión en una sola figura.
    preds = {'Random Forest': y_pred_rf, 'SVM': y_pred_svm, 'MLP': y_pred_mlp}
    """
    from sklearn.metrics import accuracy_score, f1_score
    cmaps = ["Greens", "Blues", "Oranges"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Matrices de Confusión — Mejores Modelos", fontsize=14, fontweight="bold")

    for ax, (nombre, y_pred), cmap in zip(axes, preds.items(), cmaps):
        cm = get_confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(ax=ax, colorbar=False, cmap=cmap)
        acc = accuracy_score(y_test, y_pred)
        f1  = f1_score(y_test, y_pred, average="weighted")
        ax.set_title(f"{nombre}\nAcc={acc:.4f} | F1={f1:.4f}", fontweight="bold")

    plt.tight_layout()
    ensure_parent_dir(save_path)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_metrics_bar(results: dict, save_path: str):
    """
    Barplot comparativo de Accuracy y F1 para los 3 modelos, con valores anotados.
    results = {nombre_modelo: {'accuracy': v, 'f1': v, ...}}
    """
    tecncias = list(results.keys())
    accuracies = [results[t]["accuracy"] for t in tecncias]
    f1s = [results[t]["f1"] for t in tecncias]

    x = np.arange(len(tecncias))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    bars1 = ax.bar(x - width / 2, accuracies, width, label="Accuracy", color="#2196F3", alpha=0.85)
    bars2 = ax.bar(x + width / 2, f1s,        width, label="F1-Score",  color="#FF9800", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(tecncias, fontsize=12)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score")
    ax.set_title("Comparación de Técnicas — Mejor Configuración", fontweight="bold")
    ax.legend()
    ax.yaxis.grid(True, alpha=0.3)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{bar.get_height():.4f}", ha="center", va="bottom", fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{bar.get_height():.4f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    ensure_parent_dir(save_path)
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_feature_importance(model, feature_names, save_path: str, top_n: int = 12):
    """Top N features más importantes del Random Forest."""
    feat_imp = pd.Series(model.feature_importances_, index=feature_names)
    feat_imp = feat_imp.sort_values(ascending=False).head(top_n)
    fig, ax = plt.subplots(figsize=(10, 5))
    feat_imp.plot(kind="bar", ax=ax, color="#4CAF50", edgecolor="black")
    ax.set_title(f"Top {top_n} Variables más Importantes — Random Forest", fontweight="bold")
    ax.set_ylabel("Importancia")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    ensure_parent_dir(save_path)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_mlp_loss(mlp_model, save_path: str):
    """Curva de pérdida por época del MLP."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(mlp_model.loss_curve_, color="darkorange", label="Train loss")
    if hasattr(mlp_model, "validation_scores_") and mlp_model.validation_scores_:
        ax.plot(mlp_model.validation_scores_, color="steelblue", linestyle="--", label="Val score")
    ax.set_title("Curva de pérdida — MLP")
    ax.set_xlabel("Época")
    ax.set_ylabel("Loss / Score")
    ax.legend()
    plt.tight_layout()
    ensure_parent_dir(save_path)
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_eda_combined(df, target_col: str, save_path: str):
    """
    Figura combinada de 6 subplots para el EDA:
    target, sexo x clase, ciclovital x clase, edad, factores de riesgo, casos por año.
    """
    factores = [
        "enfermedades_dolorosas", "maltrato_sexual", "muerte_familiar",
        "conflicto_pareja", "problemas_economicos",
        "problemas_juridicos", "problemas_laborales", "suicidio_amigo"
    ]
    factores_exist = [f for f in factores if f in df.columns]
    clases = df[target_col].unique()
    colors = ["#2196F3", "#FF5722"]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("Exploración de Variables — Conducta Suicida Bogotá", fontsize=14, fontweight="bold")

    # 1. Distribución del target
    conteo = df[target_col].value_counts()
    axes[0, 0].bar(conteo.index, conteo.values, color=colors, edgecolor="black")
    axes[0, 0].set_title("Variable Objetivo")
    axes[0, 0].set_ylabel("Casos")
    axes[0, 0].tick_params(axis="x", rotation=15)
    for i, v in enumerate(conteo.values):
        axes[0, 0].text(i, v + 500, f"{v:,}", ha="center", fontweight="bold")

    # 2. Sexo x clase
    if "sexo" in df.columns:
        sexo_clase = df.groupby(["sexo", target_col]).size().unstack()
        sexo_clase.plot(kind="bar", ax=axes[0, 1], color=colors)
        axes[0, 1].set_title("Distribución por Sexo")
        axes[0, 1].tick_params(axis="x", rotation=0)
        axes[0, 1].legend(fontsize=7)

    # 3. Ciclo vital x clase
    if "ciclovital" in df.columns:
        ciclo_clase = df.groupby(["ciclovital", target_col]).size().unstack()
        ciclo_clase.plot(kind="bar", ax=axes[0, 2], color=colors)
        axes[0, 2].set_title("Distribución por Ciclo Vital")
        axes[0, 2].tick_params(axis="x", rotation=30)
        axes[0, 2].legend(fontsize=7)

    # 4. Distribución de edad por clase
    if "edad" in df.columns:
        for clase, color in zip(df[target_col].unique(), colors):
            df[df[target_col] == clase]["edad"].plot(
                kind="hist", ax=axes[1, 0], alpha=0.6, bins=30, color=color, label=clase)
        axes[1, 0].set_title("Distribución de Edad")
        axes[1, 0].set_xlabel("Edad")
        axes[1, 0].legend(fontsize=7)

    # 5. Frecuencia de factores de riesgo
    if factores_exist:
        freq = df[factores_exist].sum().sort_values(ascending=True)
        freq.plot(kind="barh", ax=axes[1, 1], color="#9C27B0")
        axes[1, 1].set_title("Frecuencia de Factores de Riesgo")
        axes[1, 1].set_xlabel("Número de casos")

    # 6. Casos por año
    if "ano_notificacion" in df.columns:
        anual = df.groupby(["ano_notificacion", target_col]).size().unstack()
        anual.plot(kind="line", ax=axes[1, 2], marker="o", color=colors)
        axes[1, 2].set_title("Casos por Año")
        axes[1, 2].set_xlabel("Año")
        axes[1, 2].legend(fontsize=7)

    plt.tight_layout()
    ensure_parent_dir(save_path)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_experiment_bars(df_rf, df_svm, df_mlp, save_path: str):
    """Barplot de accuracy por configuración de hiperparámetros para los 3 modelos."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    df_rf = df_rf.copy()
    df_rf["config"] = df_rf.apply(
        lambda r: f"n={r['n_estimators']}\nd={r['max_depth']}\ns={r['min_samples_split']}", axis=1)
    axes[0].bar(range(len(df_rf)), df_rf["Accuracy"], color="steelblue", edgecolor="black")
    axes[0].set_xticks(range(len(df_rf)))
    axes[0].set_xticklabels(df_rf["config"], fontsize=7)
    axes[0].set_title("Random Forest")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_ylim(0, 1)

    df_svm = df_svm.copy()
    df_svm["config"] = df_svm.apply(
        lambda r: f"C={r['C']}\n{r['kernel']}\ng={r['gamma']}", axis=1)
    axes[1].bar(range(len(df_svm)), df_svm["Accuracy"], color="coral", edgecolor="black")
    axes[1].set_xticks(range(len(df_svm)))
    axes[1].set_xticklabels(df_svm["config"], fontsize=7)
    axes[1].set_title("SVM")

    df_mlp = df_mlp.copy()
    df_mlp["config"] = df_mlp.apply(
        lambda r: f"{r['hidden_layers']}\n{r['activation']}\nlr={r['learning_rate']}", axis=1)
    axes[2].bar(range(len(df_mlp)), df_mlp["Accuracy"], color="mediumseagreen", edgecolor="black")
    axes[2].set_xticks(range(len(df_mlp)))
    axes[2].set_xticklabels(df_mlp["config"], fontsize=7)
    axes[2].set_title("MLP")

    plt.suptitle("Accuracy por configuración de hiperparámetros", fontsize=13, y=1.02)
    plt.tight_layout()
    ensure_parent_dir(save_path)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
