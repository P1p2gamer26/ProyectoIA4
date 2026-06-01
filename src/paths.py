"""Utilidades de rutas y creación de directorios de salida."""
import os


def ensure_parent_dir(path: str) -> None:
    """Crea el directorio padre de `path` si no existe."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def ensure_results_dirs(base: str = "results") -> None:
    """Crea la estructura estándar de carpetas en results/."""
    for sub in (
        "",
        "eda",
        "confusion_matrices",
        "training",
        "experiment",
    ):
        os.makedirs(os.path.join(base, sub) if sub else base, exist_ok=True)
