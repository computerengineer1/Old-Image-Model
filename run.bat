@echo off
cd /d "%~dp0"
title OLD MODEL IMAGES - server

rem --- memory / VRAM hygiene for a 4 GB GPU on an 8 GB system ---
set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
set HF_HOME=%~dp0models
set HF_HUB_DISABLE_TELEMETRY=1
set HF_HUB_DISABLE_IMPLICIT_TOKEN=1

if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" app.py
) else (
    echo [WARN] .venv not found - trying system Python.
    echo        If this fails, run setup.bat first.
    python app.py
)

echo.
echo Server stopped.
pause
