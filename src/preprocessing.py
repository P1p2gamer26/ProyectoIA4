"""
preprocessing.py
Carga, limpieza, codificación y split del dataset de conducta suicida.
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

TARGET_COL = "clasificaciondelaconducta"

BINARY_COLS = [
    "enfermedades_dolorosas", "maltrato_sexual", "muerte_familiar",
    "conflicto_pareja", "problemas_economicos",
    "problemas_juridicos", "problemas_laborales", "suicidio_amigo"
]

# Columnas redundantes o de demasiada granularidad que no aportan al modelo
COLS_ELIMINAR = ["codigo_localidadresidencia", "nombre_upz", "esc_educ"]

# Normalización de niveles educativos (variantes de texto en el dataset)
NIVEL_MAP = {
    "1. No fue a la escuela":                   "01_Sin_escolaridad",
    "2. Preescolar":                             "02_Preescolar",
    "3. Primaria incompleta":                    "03_Primaria_inc",
    "4. Primaria completa":                      "04_Primaria_comp",
    "5. Secundaria incompleta":                  "05_Secundaria_inc",
    "6. Secundaria completa":                    "06_Secundaria_comp",
    "7. Técnico pos secundaria incompleto":      "07_Tecnico_inc",
    "7. Técnico post-secundaria incompleta":     "07_Tecnico_inc",
    "8. Técnico post-secundaria completa":       "08_Tecnico_comp",
    "8. Técnico pos secundaria completo":        "08_Tecnico_comp",
    "9. Universidad incompleta":                 "09_Univ_inc",
    "10. Universidad completa":                  "10_Univ_comp",
    "11. Postgrado incompleto":                  "11_Postgrado_inc",
    "11. Posgrado incompleto":                   "11_Postgrado_inc",
    "12. Postgrado completo":                    "12_Postgrado_comp",
    "99. Sin dato":                              "99_Sin_dato",
}


def load_raw(path: str) -> pd.DataFrame:
    """Carga el CSV original con detección automática de separador y encoding latin-1."""
    return pd.read_csv(path, encoding="latin-1", sep=None, engine="python")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpieza del dataset:
    - Normaliza nombres de columnas
    - Elimina columnas redundantes
    - Elimina filas sin target
    - Estandariza niveleducativo
    - Rellena NaN en columnas binarias con 0
    - Rellena poblacion_diferencial con 'Ninguna'
    """
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    df = df.dropna(subset=[TARGET_COL])
    df[TARGET_COL] = df[TARGET_COL].str.strip()

    # Eliminar columnas redundantes que existan
    cols_drop = [c for c in COLS_ELIMINAR if c in df.columns]
    df = df.drop(columns=cols_drop)

    # Estandarizar niveleducativo
    if "niveleducativo" in df.columns:
        df["niveleducativo"] = df["niveleducativo"].map(NIVEL_MAP).fillna("99_Sin_dato")

    # Rellenar NaN en factores de riesgo binarios
    for col in BINARY_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # Rellenar NaN en poblacion_diferencial
    if "poblacion_diferencial" in df.columns:
        df["poblacion_diferencial"] = df["poblacion_diferencial"].fillna("Ninguna")

    return df


def encode_and_split(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Codifica variables, imputa NaN restantes, escala y divide en train/test.

    Retorna:
        X_train, X_test           — sin escalar (para RF)
        X_train_sc, X_test_sc     — escalados con StandardScaler (para SVM y MLP)
        y_train, y_test
        le                        — LabelEncoder del target
    """
    df = df.copy()

    le = LabelEncoder()
    df[TARGET_COL] = le.fit_transform(df[TARGET_COL])

    cat_cols = df.select_dtypes(include="object").columns.tolist()
    for col in cat_cols:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    # Imputar NaN restantes con la mediana
    imputer = SimpleImputer(strategy="median")
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    return X_train, X_test, X_train_sc, X_test_sc, y_train, y_test, le
