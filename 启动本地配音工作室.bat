@echo off
setlocal
cd /d "%~dp0"
chcp 65001 >nul
set "PYTHONUTF8=1"
set "PYTHONNOUSERSITE=1"
set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
if not exist "%PYTHON_EXE%" set "PYTHON_EXE=%~dp0python\python.exe"
if not exist "%PYTHON_EXE%" (
    echo Python environment missing. Run setup.bat first.
    pause
    exit /b 1
)
"%PYTHON_EXE%" "%~dp0app\app.py"
set "RESULT=%errorlevel%"
pause
exit /b %RESULT%
