@echo off
SETLOCAL

echo =========================================
echo   CCTV Incident Alert System — Startup
echo =========================================
echo.

:: ── Step 1: Create .env if missing ────────────────────────────────────────
IF NOT EXIST ".env" (
    echo [1/3] Creating .env from template...
    COPY ".env.example" ".env" > nul
    echo       .env created. Please edit it with your Twilio credentials.
    echo.
) ELSE (
    echo [1/3] .env already exists.
)

:: ── Step 2: Install API dependencies ──────────────────────────────────────
echo [2/3] Installing Python API dependencies...
IF EXIST "venv\Scripts\python.exe" (
    venv\Scripts\python.exe -m pip install -q -r requirements_api.txt
) ELSE IF EXIST "venv\Scripts\pip.exe" (
    venv\Scripts\pip install -q -r requirements_api.txt
) ELSE (
    pip install -q -r requirements_api.txt
)
echo       Done.
echo.

:: ── Step 3: Start FastAPI backend ─────────────────────────────────────────
echo [3/3] Starting FastAPI backend on http://localhost:8000 ...
echo       API Docs: http://localhost:8000/docs
echo       Press Ctrl+C to stop.
echo.

IF EXIST "venv\Scripts\python.exe" (
    venv\Scripts\python.exe -m uvicorn backend.api.main:app --host 0.0.0.0 --port 8000 --reload
) ELSE IF EXIST "venv\Scripts\uvicorn.exe" (
    venv\Scripts\uvicorn backend.api.main:app --host 0.0.0.0 --port 8000 --reload
) ELSE (
    uvicorn backend.api.main:app --host 0.0.0.0 --port 8000 --reload
)

ENDLOCAL
