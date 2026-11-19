@echo off
setlocal
cd /d "%~dp0"
chcp 65001 >nul
set "PYTHONUTF8=1"
set "PYTHONNOUSERSITE=1"
where uv >nul 2>&1
if errorlevel 1 (
    echo Install uv and reopen this script. See README.md.
    pause
    exit /b 1
)
uv venv --allow-existing --python 3.11 .venv || goto fail
set "VOICE_PYTHON=%~dp0.venv\Scripts\python.exe"
uv pip install --python "%VOICE_PYTHON%" --index https://download.pytorch.org/whl/cu128 "torch==2.8.0+cu128" "torchaudio==2.8.0+cu128" || goto fail
uv pip install --python "%VOICE_PYTHON%" -r requirements.txt || goto fail
"%VOICE_PYTHON%" app\download_models.py || goto fail
echo Setup complete. Run the launcher or environment check.
pause
exit /b 0

:fail
echo Setup stopped. Check the error above and rerun setup.bat.
pause
exit /b 1
