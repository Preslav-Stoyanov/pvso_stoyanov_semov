@echo off
REM run.bat - Start Gaussian Splatting container on Windows

echo === Gaussian Splatting Docker ===
echo.

docker version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Docker is not running or not installed.
    pause
    exit /b 1
)

docker image inspect gaussian-splatting:latest >nul 2>&1
if errorlevel 1 (
    echo Image not found. Building now, this takes 30-60 min...
    docker compose build
    if errorlevel 1 (
        echo ERROR: Build failed!
        pause
        exit /b 1
    )
)

echo Starting container...
echo.
echo For GUI on Windows 10: install VcXsrv and start it first
echo.

docker compose run --rm -e DISPLAY=host.docker.internal:0 gaussian-splatting

pause
