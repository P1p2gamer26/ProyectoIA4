"""
verification.py
Verificación de calidad de datos antes del entrenamiento.
"""
import pandas as pd

TARGET_COL = "clasificaciondelaconducta"
VALID_TARGETS = {"Ideación suicida", "Intento de Suicidio"}


def _target_series(df: pd.DataFrame) -> pd.Series | None:
    """Localiza la columna target con o sin normalización de nombres."""
    if TARGET_COL in df.columns:
        return df[TARGET_COL]
    lower_map = {c.lower().strip().replace(" ", "_"): c for c in df.columns}
    if TARGET_COL in lower_map:
        return df[lower_map[TARGET_COL]]
    return None


def verify_raw(df: pd.DataFrame) -> dict:
    """
    Ejecuta chequeos de calidad sobre el dataset crudo (antes de clean).

    Retorna un diccionario con métricas y mensajes para reporte / notebook.
    """
    report = {}
    target = _target_series(df)

    # Duplicados
    n_dup = int(df.duplicated().sum())
    report["duplicados"] = n_dup
    report["duplicados_msg"] = (
        f"Filas duplicadas exactas: {n_dup:,}"
        + (" — se eliminarán en limpieza." if n_dup else " — ninguna detectada.")
    )

    # Target
    if target is not None:
        target_vals = set(target.dropna().astype(str).str.strip().unique())
        invalid = target_vals - VALID_TARGETS
        report["target_valores"] = sorted(target_vals)
        report["target_invalidos"] = sorted(invalid)
        report["target_msg"] = (
            f"Valores de target: {sorted(target_vals)}"
            + (f" | Inválidos: {sorted(invalid)}" if invalid else " — todos válidos.")
        )
        report["sin_target"] = int(target.isna().sum())

    # Edad — outliers por IQR
    if "edad" in df.columns:
        edad = pd.to_numeric(df["edad"], errors="coerce")
        q1, q3 = edad.quantile(0.25), edad.quantile(0.75)
        iqr = q3 - q1
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outliers = ((edad < low) | (edad > high)) & edad.notna()
        report["edad_min"] = float(edad.min())
        report["edad_max"] = float(edad.max())
        report["edad_outliers_iqr"] = int(outliers.sum())
        report["edad_msg"] = (
            f"Edad: min={report['edad_min']:.0f}, max={report['edad_max']:.0f}, "
            f"outliers IQR={report['edad_outliers_iqr']:,} (no se eliminan; RF es robusto)."
        )

    # Tipos
    report["tipos_numericos"] = df.select_dtypes(include="number").columns.tolist()
    report["tipos_object"] = df.select_dtypes(include="object").columns.tolist()

    return report


def print_verification_report(report: dict) -> None:
    """Imprime el reporte de verificación de forma legible."""
    print("=== Verificación de datos ===\n")
    for key in ("duplicados_msg", "target_msg", "edad_msg"):
        if key in report:
            print(report[key])
    if report.get("sin_target"):
        print(f"Filas sin target (se eliminarán): {report['sin_target']:,}")
    print(f"\nColumnas numéricas: {len(report.get('tipos_numericos', []))}")
    print(f"Columnas categóricas/texto: {len(report.get('tipos_object', []))}")
