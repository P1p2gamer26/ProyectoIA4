# setup.ps1 — ejecutar una sola vez despues de git clone
# Crea el entorno virtual, instala dependencias y registra el kernel de Jupyter

Write-Host "`n=== Setup ProyectoIA4 ===" -ForegroundColor Cyan

# 0. Verificar que Python este instalado
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host "`n[ERROR] Python no encontrado." -ForegroundColor Red
    Write-Host "Descargalo desde: https://www.python.org/downloads/" -ForegroundColor Yellow
    Write-Host "Asegurate de marcar 'Add Python to PATH' durante la instalacion." -ForegroundColor Yellow
    exit 1
}
$pyVersion = python --version
Write-Host "`nPython detectado: $pyVersion" -ForegroundColor Green

# 1. Crear el venv si no existe
if (-not (Test-Path ".venv")) {
    Write-Host "`n[1/3] Creando entorno virtual..." -ForegroundColor Yellow
    python -m venv .venv
} else {
    Write-Host "`n[1/3] Entorno virtual ya existe, saltando..." -ForegroundColor Green
}

# 2. Instalar dependencias
Write-Host "`n[2/3] Instalando dependencias..." -ForegroundColor Yellow
.venv\Scripts\python.exe -m pip install --upgrade pip --quiet
.venv\Scripts\python.exe -m pip install -r requirements.txt

# 3. Registrar el kernel para que VS Code lo detecte automaticamente
Write-Host "`n[3/3] Registrando kernel de Jupyter..." -ForegroundColor Yellow
.venv\Scripts\python.exe -m ipykernel install --user --name proyectoIA4 --display-name "Python (ProyectoIA4)"

Write-Host "`n=== Listo ===" -ForegroundColor Green
Write-Host "Abre VS Code, selecciona el kernel 'Python (ProyectoIA4)' y ejecuta los notebooks en orden:" -ForegroundColor Cyan
Write-Host "  1. notebooks/01_eda_limpieza.ipynb"
Write-Host "  2. notebooks/02_modelos.ipynb"
Write-Host "  3. notebooks/03_experimento.ipynb"
