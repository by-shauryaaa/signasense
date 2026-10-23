@echo off
setlocal
echo ============================================================
echo   SignaSense — Pretrained 250 ASL Signs Video Recognition
echo ============================================================
set PYTHONIOENCODING=utf-8

if "%~1"=="" goto :no_video

echo [INFO] Processing: %1
echo.

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
    echo [ERROR] Video inference stopped with code %ERRORLEVEL%.
)
echo.
pause
exit /b 0

:no_video
echo [INFO] No video file specified.
echo.
echo How to use:
echo   1. Drag and drop any video file (.mp4, .mov, .avi) directly onto this .bat file.
echo   2. Or run from terminal:
echo      run_pretrained_video.bat path\to\asl_video.mp4
echo.
pause
exit /b 1
