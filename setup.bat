@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"
title OLD MODEL IMAGES - Setup

echo ============================================================
echo    OLD MODEL IMAGES  -  one-time setup
echo    Target: Windows 11 / RTX 2050 4GB / Ryzen 5 7535HS / 8GB RAM
echo    Downloads roughly 4-5 GB total. Stable connection required.
echo ============================================================
echo.

rem ---------------- 1. locate a usable 64-bit Python 3.10-3.12 ----------------
set "PYCMD="
for %%V in (3.12 3.11 3.10) do (
    if not defined PYCMD (
        py -%%V -c "import sys; sys.exit(0 if sys.maxsize > 2**32 else 1)" >nul 2>&1
        if not errorlevel 1 set "PYCMD=py -%%V"
    )
)
if not defined PYCMD (
    python -c "import sys; v=sys.version_info[:2]; sys.exit(0 if (3,10) <= v <= (3,12) and sys.maxsize > 2**32 else 1)" >nul 2>&1
    if not errorlevel 1 set "PYCMD=python"
)

if not defined PYCMD (
    echo [ERROR] No 64-bit Python 3.10 / 3.11 / 3.12 found on PATH.
    echo.
    where winget >nul 2>&1
    if not errorlevel 1 (
        choice /c YN /m "winget detected - install Python 3.11 now? [Y/N]"
        if !errorlevel! equ 1 (
            winget install -e --id Python.Python.3.11
            echo.
            echo Python installation finished. CLOSE this window, open a NEW
            echo terminal, and run setup.bat again so the new Python is on PATH.
            pause
            exit /b 1
        )
    )
    echo Install Python 3.11 from https://www.python.org/downloads/windows/
    echo IMPORTANT: tick "Add python.exe to PATH" in the installer, then
    echo re-run setup.bat.
    pause
    exit /b 1
)

echo [1/5] Python found:
%PYCMD% --version
%PYCMD% -c "import struct; print('      architecture:', struct.calcsize('P')*8, 'bit')"
echo.

rem ---------------- 2. virtual environment ----------------
if exist ".venv\Scripts\python.exe" (
    echo [2/5] Virtual environment .venv already exists - reusing it.
) else (
    echo [2/5] Creating virtual environment .venv ...
    %PYCMD% -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Could not create the virtual environment.
        pause
        exit /b 1
    )
)
set "VPY=.venv\Scripts\python.exe"
echo.

rem ---------------- 3. pinned dependencies ----------------
echo [3/5] Upgrading pip tooling...
%VPY% -m pip install --upgrade "pip==24.0" setuptools wheel
if errorlevel 1 (
    echo [ERROR] pip bootstrap failed. Check your internet connection.
    pause
    exit /b 1
)
echo.
echo [4/5] Installing pinned dependencies - torch 2.2.1+cu121, diffusers stack...
echo       This is the big download (~2.5 GB). Do not close this window.
%VPY% -m pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo [ERROR] Dependency installation failed.
    echo   Common fixes:
    echo     - check internet connection / proxy settings
    echo     - temporarily pause antivirus, re-run setup.bat
    echo     - make sure you have ~10 GB free disk space
    pause
    exit /b 1
)
echo.

rem ---------------- optional xformers ----------------
echo       Trying optional xformers 0.0.25 (memory-efficient attention)...
%VPY% -m pip install xformers==0.0.25 --index-url https://download.pytorch.org/whl/cu121 >nul 2>&1
if errorlevel 1 (
    echo [WARN] xformers not installed. That is OK - the app automatically
    echo        falls back to PyTorch SDPA attention, which is nearly as fast.
) else (
    echo       xformers installed successfully.
)
echo.

rem ---------------- 4. verify torch + CUDA ----------------
echo [5/5] Verifying PyTorch CUDA...
%VPY% -c "import torch; print('       torch', torch.__version__, '| CUDA build', torch.version.cuda); print('       GPU visible:', torch.cuda.is_available()); print('       GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE - generation will not work!')"
echo.

rem ---------------- 5. model weights ----------------
echo Downloading model weights (DreamShaper 8 fp16 + Real-ESRGAN + LoRAs)...
echo This fetches ~2.5 GB from Hugging Face. You can re-run this step any
echo time with:  .venv\Scripts\python.exe download_models.py
%VPY% download_models.py
if errorlevel 1 (
    echo.
    echo [WARN] Model download did not complete. The app will still start,
    echo        but generation needs the checkpoint. Re-run:
    echo          .venv\Scripts\python.exe download_models.py
    echo        See logs\download_report.txt for details.
)

echo.
echo ============================================================
echo    SETUP COMPLETE
echo.
echo    Start the app any time with:   run.bat
echo    It will open http://127.0.0.1:8756 in your browser.
echo    First generation loads the model and takes ~1-2 minutes;
echo    later ones are much faster.
echo ============================================================
pause
endlocal
