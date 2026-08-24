@echo off
cd /d "%~dp0"
echo Starting Sleeper Trade Finder...
python -m streamlit run app.py
if errorlevel 1 (
    echo.
    echo Streamlit did not start successfully.
    echo Try running setup_windows.bat first.
    pause
)
