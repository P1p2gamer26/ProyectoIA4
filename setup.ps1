# setup.ps1 — atajo Windows (llama a bootstrap.py)
# También puedes hacer doble clic en setup.bat

Write-Host "`n=== Setup ProyectoIA4 ===" -ForegroundColor Cyan
Set-Location $PSScriptRoot

foreach ($try in @("py -3.12", "py -3.11", "py -3.10", "python")) {
    try {
        Invoke-Expression "$try scripts\bootstrap.py"
        if ($LASTEXITCODE -eq 0) { exit 0 }
    } catch { }
}

Write-Host "`n[ERROR] Ejecuta: py -3.11 scripts\bootstrap.py" -ForegroundColor Red
exit 1
