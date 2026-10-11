@echo off
SETLOCAL

echo =========================================
echo   CCTV Incident Alert — React Frontend
echo =========================================
echo.

WHERE node >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo ERROR: Node.js is not installed or not in PATH.
    echo.
    echo Please install Node.js from: https://nodejs.org/
    echo Then re-run this script.
    pause
    exit /b 1
)

echo Node.js found: 
node --version

IF NOT EXIST "node_modules" (
    echo.
    echo Installing React dependencies...
    npm install
)

echo.
echo Starting React dev server on http://localhost:5173 ...
echo.
npm run dev

ENDLOCAL
