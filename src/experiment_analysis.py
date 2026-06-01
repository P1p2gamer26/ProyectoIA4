"""
experiment_analysis.py
Análisis comparativo automático a partir de los resultados del experimento.
"""
import ast
import pandas as pd


def _accuracy_spread(df: pd.DataFrame) -> float:
    return float(df["Accuracy"].max() - df["Accuracy"].min())


def _most_influential_param(df: pd.DataFrame, param_cols: list) -> tuple[str, float]:
    """Estima el hiperparámetro con mayor efecto medio sobre Accuracy."""
    best_col, best_range = param_cols[0], 0.0
    for col in param_cols:
        means = df.groupby(col)["Accuracy"].mean()
        spread = float(means.max() - means.min())
        if spread > best_range:
            best_range = spread
            best_col = col
    return best_col, best_range


def generate_comparative_analysis(
    df_rf: pd.DataFrame,
    df_svm: pd.DataFrame,
    df_mlp: pd.DataFrame,
    svm_sample_size: int,
    train_size: int,
) -> str:
    """
    Genera el análisis comparativo en texto (markdown) para notebook y PPT.
    """
    best_acc_rf = df_rf.loc[df_rf["Accuracy"].idxmax()]
    best_acc_svm = df_svm.loc[df_svm["Accuracy"].idxmax()]
    best_acc_mlp = df_mlp.loc[df_mlp["Accuracy"].idxmax()]

    all_best = [
        ("Random Forest", float(df_rf["F1 (weighted)"].max()), float(df_rf["Accuracy"].max())),
        ("SVM", float(df_svm["F1 (weighted)"].max()), float(df_svm["Accuracy"].max())),
        ("MLP", float(df_mlp["F1 (weighted)"].max()), float(df_mlp["Accuracy"].max())),
    ]
    best_f1_tech = max(all_best, key=lambda x: x[1])[0]

    # Mejor accuracy global entre las 8 configs de cada técnica (mejor run por técnica)
    global_acc = {
        "Random Forest": best_acc_rf["Accuracy"],
        "SVM": best_acc_svm["Accuracy"],
        "MLP": best_acc_mlp["Accuracy"],
    }
    best_acc_tech = max(global_acc, key=global_acc.get)

    spreads = {
        "Random Forest": _accuracy_spread(df_rf),
        "SVM": _accuracy_spread(df_svm),
        "MLP": _accuracy_spread(df_mlp),
    }
    most_sensitive = max(spreads, key=spreads.get)

    inf_rf = _most_influential_param(df_rf, ["n_estimators", "max_depth", "min_samples_split"])
    inf_svm = _most_influential_param(df_svm, ["C", "kernel", "gamma"])
    inf_mlp = _most_influential_param(
        df_mlp, ["hidden_layers", "activation", "learning_rate"]
    )

    lines = [
        "## Análisis comparativo del experimento",
        "",
        f"**Mejor técnica en accuracy (mejor corrida por técnica):** {best_acc_tech} "
        f"(RF={global_acc['Random Forest']:.4f}, SVM={global_acc['SVM']:.4f}, "
        f"MLP={global_acc['MLP']:.4f}).",
        "",
        f"**Mejor técnica en F1 ponderado (criterio principal por desbalance 73/27):** "
        f"{best_f1_tech}.",
        "",
        f"**Técnica más sensible a hiperparámetros** (mayor rango de Accuracy entre las "
        f"8 configuraciones): **{most_sensitive}** "
        f"(ΔAcc ≈ {spreads[most_sensitive]:.4f}). "
        f"Rangos: RF={spreads['Random Forest']:.4f}, SVM={spreads['SVM']:.4f}, "
        f"MLP={spreads['MLP']:.4f}.",
        "",
        "**Hiperparámetro más influyente por técnica** (mayor diferencia de Accuracy "
        "promedio al variar solo ese parámetro):",
        f"- RF: `{inf_rf[0]}` (ΔAcc promedio entre niveles ≈ {inf_rf[1]:.4f})",
        f"- SVM: `{inf_svm[0]}` (Δ ≈ {inf_svm[1]:.4f})",
        f"- MLP: `{inf_mlp[0]}` (Δ ≈ {inf_mlp[1]:.4f})",
        "",
        "**Nota metodológica:** SVM se entrenó con muestra estratificada de "
        f"{svm_sample_size:,} registros (de {train_size:,} en train) por costo "
        "computacional O(n²); RF y MLP usaron el train completo. La comparación "
        "entre técnicas debe interpretarse considerando esta diferencia.",
        "",
        "**Conclusión general:**",
        f"- {best_acc_tech} obtuvo la mayor accuracy en el experimento; {best_f1_tech} "
        "destaca en F1 ponderado, más adecuado con clases desbalanceadas.",
        f"- {most_sensitive} mostró mayor variabilidad ante cambios de hiperparámetros.",
        "- El diseño factorial 2³ permitió comparar 24 configuraciones de forma "
        "sistemática sin ANOVA, identificando combinaciones favorables para el PPT "
        "y la presentación en clase.",
    ]
    return "\n".join(lines)


def parse_mlp_layers(s: str) -> tuple:
    """Convierte string '(100, 50)' a tupla para MLPClassifier."""
    return tuple(ast.literal_eval(s))
