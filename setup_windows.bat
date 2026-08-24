@echo off
cd /d "%~dp0"
echo ==========================================
echo Sleeper Trade Finder - Windows Setup
echo ==========================================
echo.
echo Installing/updating required packages using the SAME Python interpreter...
python -m pip install -r requirements.txt

if errorlevel 1 (
    echo.
    echo Package installation failed.
    echo Run: python --version
    echo and: python -m pip --version
    pause
    exit /b 1
)

echo.
echo Packages installed successfully.
echo Starting the app...
python -m streamlit run app.py

if errorlevel 1 (
    echo.
    echo The app did not start.
    echo Please copy the error shown above.
    pause
)
