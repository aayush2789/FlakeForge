@echo off
REM FlakeForge Inference - Windows Quick Start Batch Script

echo.
echo ============================================================
echo   FlakeForge Inference - Windows Quick Start
echo ============================================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Please install Python 3.10+
    echo Visit: https://www.python.org/downloads/
    pause
    exit /b 1
)

echo [1/3] Python Check: OK

REM Check if NVIDIA_API_KEY is set
if "%NVIDIA_API_KEY%"=="" (
    echo.
    echo WARNING: NVIDIA_API_KEY not set!
    echo.
    echo Please set it before running:
    echo   set NVIDIA_API_KEY=your-key-here
    echo.
    echo Or create a .env file with your API key.
    echo.
    set /p CONTINUE="Continue anyway? (y/n): "
    if /i not "%CONTINUE%"=="y" exit /b 1
)

echo [2/3] API Key Check: OK (or skipped)

REM Run the validation script
echo.
echo [3/3] Running validation...
echo.
python QUICKSTART.py

echo.
echo ============================================================
echo   Next Steps
echo ============================================================
echo.
echo To run inference, choose one:
echo.
echo Option A - Show Test Flakiness (quick, 30 seconds):
echo   cd test_repos\timing_race_minimal
echo   python test_runner.py --show-flakiness --runs 20
echo.
echo Option B - Full Inference (requires running server):
echo   1. Start server in another terminal:
echo      uv run server --port 8000
echo.
echo   2. Then run inference:
echo      set ENV_BASE_URL=http://localhost:8000
echo      cd test_repos\timing_race_minimal
echo      python test_runner.py --run-inference --steps 5
echo.
echo Results saved to: outputs\flakeforge_summary_*.json
echo.
pause
