# SignaSense — PowerShell Launcher
# Automatically runs using the active Python environment or dedicated signasense conda env

$pythonExe = "python"

if (Get-Command "python" -ErrorAction SilentlyContinue) {
    $pythonExe = "python"
} elseif (Test-Path "$env:USERPROFILE\miniconda3\envs\signasense\python.exe") {
    $pythonExe = "$env:USERPROFILE\miniconda3\envs\signasense\python.exe"
} elseif (Test-Path "$env:USERPROFILE\anaconda3\envs\signasense\python.exe") {
    $pythonExe = "$env:USERPROFILE\anaconda3\envs\signasense\python.exe"
} elseif (Test-Path "C:\Users\Ankit Singh Tomar\miniconda3\envs\signasense\python.exe") {
    $pythonExe = "C:\Users\Ankit Singh Tomar\miniconda3\envs\signasense\python.exe"
} else {
    Write-Host "[ERROR] Python not found in PATH or signasense conda environment." -ForegroundColor Red
    Write-Host "Please activate your environment: conda activate signasense" -ForegroundColor Yellow
    exit 1
}

$scriptPath = Join-Path $PSScriptRoot "src\inference_pretrained.py"

& $pythonExe $scriptPath @args
