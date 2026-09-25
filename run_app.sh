#!/usr/bin/env bash
# One-click setup and launch for Mac/Linux
cd "$(dirname "$0")"
if [ ! -d venv ]; then
  echo "Creating virtual environment..."
  python3 -m venv venv
fi
source venv/bin/activate
echo "Installing requirements (first time only takes a few minutes)..."
pip install -q -r requirements.txt
echo "Starting the app... it will open in your browser."
streamlit run app.py
