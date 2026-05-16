@echo off
setlocal

echo ============================================================
echo  VoiceClone .exe Builder
echo ============================================================
echo.

REM Detect Python command (prefer venv, then py, python, python3)
set PYTHON=
if exist venv311\Scripts\python.exe (
    set PYTHON=venv311\Scripts\python.exe
    echo Using venv311 Python.
    goto :found_python
)
py --version >nul 2>&1
if not errorlevel 1 ( set PYTHON=py & goto :found_python )
python --version >nul 2>&1
if not errorlevel 1 ( set PYTHON=python & goto :found_python )
python3 --version >nul 2>&1
if not errorlevel 1 ( set PYTHON=python3 & goto :found_python )
echo ERROR: No Python found. Install Python 3.11 and try again.
pause
exit /b 1
:found_python
echo Python command: %PYTHON%
echo.

REM Step 1: ensure PyInstaller is installed
%PYTHON% -m pip show pyinstaller >nul 2>&1
if errorlevel 1 (
    echo Installing PyInstaller...
    %PYTHON% -m pip install pyinstaller
)

REM Step 2: download ffmpeg.exe if not present
if not exist ffmpeg.exe (
    echo Downloading ffmpeg.exe...
    %PYTHON% download_ffmpeg.py
    if errorlevel 1 (
        echo ERROR: ffmpeg download failed. Aborting.
        pause
        exit /b 1
    )
) else (
    echo ffmpeg.exe already present, skipping download.
)

echo.
REM Step 3: run PyInstaller
echo Building with PyInstaller...
%PYTHON% -m PyInstaller voiceclone.spec --clean --noconfirm

if errorlevel 1 (
    echo.
    echo BUILD FAILED. Check output above for errors.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo  Build complete!
echo  Output: dist\VoiceClone\VoiceClone.exe
echo ============================================================
pause
