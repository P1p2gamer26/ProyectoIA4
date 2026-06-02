#!/usr/bin/env python3
"""
Configuración automática del proyecto (Windows / macOS / Linux).

Uso (desde la raíz del repo):
    python scripts/bootstrap.py
    py -3.11 scripts/bootstrap.py          # Windows si 'python' es 3.7

Crea .venv, instala dependencias, registra kernel Jupyter y carpetas results/.
"""
from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

MIN_PYTHON = (3, 10)
ROOT = Path(__file__).resolve().parents[1]
VENV_DIR = ROOT / ".venv"
REQUIREMENTS = ROOT / "requirements.txt"
KERNEL_NAME = "proyectoIA4"
KERNEL_DISPLAY = "Python (ProyectoIA4)"


def find_python() -> list[list[str]]:
    """Candidatos [ejecutable, ...args iniciales] con Python >= 3.10."""
    candidates: list[list[str]] = []

    if sys.version_info >= MIN_PYTHON:
        candidates.append([sys.executable])

    if platform.system() == "Windows":
        for ver in ("3.12", "3.11", "3.10"):
            candidates.append(["py", f"-{ver}"])
    else:
        for name in ("python3.12", "python3.11", "python3.10", "python3", "python"):
            candidates.append([name])

    return candidates


def run_cmd(argv: list[str], *, cwd: Path | None = None) -> None:
    print(f"  → {' '.join(argv)}")
    subprocess.run(argv, cwd=cwd or ROOT, check=True)


def version_ok(argv: list[str]) -> bool:
    try:
        out = subprocess.check_output(
            [*argv, "--version"],
            stderr=subprocess.STDOUT,
            text=True,
            cwd=ROOT,
        )
        # "Python 3.11.5"
        for token in out.strip().replace(",", "").split():
            clean = token.replace("Python", "")
            if clean and clean[0].isdigit():
                parts = clean.split(".")
                major, minor = int(parts[0]), int(parts[1])
                return major == 3 and minor >= MIN_PYTHON[1]
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError, IndexError):
        pass
    return False


def resolve_python() -> list[str]:
    for argv in find_python():
        if version_ok(argv):
            return argv
    return []


def venv_python() -> Path:
    if platform.system() == "Windows":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def main() -> int:
    print("\n=== ProyectoIA4 — configuración automática ===\n")

    py = resolve_python()
    if not py:
        print("[ERROR] No se encontró Python 3.10 o superior.")
        print("  Windows: instala desde https://www.python.org/downloads/")
        print("  Luego ejecuta:  py -3.11 scripts/bootstrap.py")
        return 1

    ver = subprocess.check_output([*py, "--version"], text=True, cwd=ROOT).strip()
    print(f"Python usado: {ver}\n")

    if not VENV_DIR.exists():
        print("[1/4] Creando entorno virtual .venv ...")
        run_cmd(py + ["-m", "venv", str(VENV_DIR)])
    else:
        print("[1/4] .venv ya existe.")

    vpy = venv_python()
    if not vpy.exists():
        print(f"[ERROR] No se creó {vpy}")
        return 1

    print("\n[2/4] Instalando dependencias ...")
    run_cmd([str(vpy), "-m", "pip", "install", "--upgrade", "pip", "-q"])
    run_cmd([str(vpy), "-m", "pip", "install", "-r", str(REQUIREMENTS)])

    print("\n[3/4] Registrando kernel Jupyter ...")
    run_cmd([
        str(vpy), "-m", "ipykernel", "install",
        "--user", "--name", KERNEL_NAME,
        "--display-name", KERNEL_DISPLAY,
    ])

    print("\n[4/4] Creando carpetas results/ ...")
    for sub in ("", "eda", "confusion_matrices", "training", "experiment"):
        (ROOT / "results" / sub).mkdir(parents=True, exist_ok=True)
    (ROOT / "presentacion").mkdir(exist_ok=True)

    print("\n=== Listo ===")
    print(f"\nIntérprete del proyecto:\n  {vpy}")
    print(f"\nEn Cursor/VS Code → kernel → Python Environments →\n  .venv  o  {KERNEL_DISPLAY}")
    print("\nNotebooks en orden:")
    print("  1. notebooks/01_eda_limpieza.ipynb")
    print("  2. notebooks/03_experimento.ipynb")
    print("  3. notebooks/02_modelos.ipynb")
    print("\nO por terminal:")
    if platform.system() == "Windows":
        print("  .venv\\Scripts\\python.exe scripts\\run_experiment.py")
    else:
        print("  .venv/bin/python scripts/run_experiment.py")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
