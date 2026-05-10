@echo off
setlocal

echo === Creating Python virtual environment ===
python -m venv .venv
if errorlevel 1 (
    echo ERROR: Failed to create virtual environment.
    echo Make sure Python 3.10+ is installed: https://www.python.org/downloads/
    pause & exit /b 1
)

echo === Installing PyTorch 2.6 with CUDA 12.4 ===
.venv\Scripts\pip install torch==2.6.0+cu124 torchaudio==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124
if errorlevel 1 (
    echo ERROR: Failed to install PyTorch.
    pause & exit /b 1
)

echo === Installing Python dependencies ===
.venv\Scripts\pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install Python dependencies.
    pause & exit /b 1
)

echo === Building frontend ===
cd frontend
call npm install
if errorlevel 1 (
    echo ERROR: Failed to install Node.js packages.
    echo Make sure Node.js is installed: https://nodejs.org/
    cd ..
    pause & exit /b 1
)
call npm run build
if errorlevel 1 (
    echo ERROR: Frontend build failed.
    cd ..
    pause & exit /b 1
)
cd ..

echo.
echo === Setup complete! Run run.bat to start. ===
echo.
echo NOTE: ffmpeg must be on your PATH for MP3 export.
echo Download: https://ffmpeg.org/download.html
pause
