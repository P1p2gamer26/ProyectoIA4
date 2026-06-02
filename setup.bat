@echo off
REM Doble clic o: setup.bat  — configura .venv en Windows
cd /d "%~dp0"
echo.
echo === ProyectoIA4 Setup (Windows) ===
echo.

where py >nul 2>&1
if %errorlevel%==0 (
    py -3.12 scripts\bootstrap.py 2>nul && goto :done
    py -3.11 scripts\bootstrap.py 2>nul && goto :done
    py -3.10 scripts\bootstrap.py 2>nul && goto :done
)

where python >nul 2>&1
if %errorlevel%==0 (
    python scripts\bootstrap.py 2>nul && goto :done
)

echo [ERROR] Instala Python 3.10+ desde https://www.python.org/downloads/
echo Marca "Add Python to PATH" y vuelve a ejecutar setup.bat
pause
exit /b 1

:done
echo.
pause
exit /b 0
