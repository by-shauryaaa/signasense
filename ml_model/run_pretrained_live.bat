@echo off
setlocal
echo ============================================================
echo   SignaSense — Pretrained 250 ASL Signs Live Webcam Recognition
echo ============================================================
set PYTHONIOENCODING=utf-8

set "PYTHON_CMD=python"
python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    if exist "%USERPROFILE%\miniconda3\envs\signasense\python.exe" (
        set "PYTHON_CMD=%USERPROFILE%\miniconda3\envs\signasense\python.exe"
    ) else if exist "%USERPROFILE%\anaconda3\envs\signasense\python.exe" (
        set "PYTHON_CMD=%USERPROFILE%\anaconda3\envs\signasense\python.exe"
    ) else if exist "C:\Users\Ankit Singh Tomar\miniconda3\envs\signasense\python.exe" (
        set "PYTHON_CMD=C:\Users\Ankit Singh Tomar\miniconda3\envs\signasense\python.exe"
    ) else (
        echo [ERROR] Python not found in PATH or signasense conda environment.
        echo Please activate your environment: conda activate signasense
        pause
        exit /b 1
    )
)

"%PYTHON_CMD%" src\inference_pretrained.py %*
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Pretrained live inference stopped with code %ERRORLEVEL%.
)
pause
