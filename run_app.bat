@echo off
REM One-click setup and launch for Windows
cd /d "%~dp0"
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
)
call venv\Scripts\activate
echo Installing requirements (first time only takes a few minutes)...
pip install -q -r requirements.txt
echo Starting the app... it will open in your browser.
streamlit run app.py
pause
