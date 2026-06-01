"""
evaluation.py
Funciones de evaluación de modelos: métricas y matriz de confusión.
"""
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix
)


def evaluate(model, X_test, y_test, le=None):
    """
    Evalúa un modelo y retorna predicciones, métricas y reporte completo.

    Retorna:
        y_pred   — predicciones
        metrics  — dict con accuracy, f1, precision, recall (weighted)
        report   — classification_report en texto
    """
    y_pred = model.predict(X_test)
    labels = le.classes_ if le is not None else None
    report = classification_report(y_test, y_pred, target_names=labels)
    metrics = {
        "accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "f1":        round(f1_score(y_test, y_pred, average="weighted"), 4),
        "precision": round(precision_score(y_test, y_pred, average="weighted"), 4),
        "recall":    round(recall_score(y_test, y_pred, average="weighted"), 4),
    }
    return y_pred, metrics, report


def get_confusion_matrix(y_test, y_pred):
    """Retorna la matriz de confusión como array numpy."""
    return confusion_matrix(y_test, y_pred)
