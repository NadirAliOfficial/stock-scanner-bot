@echo off
cd /d "%~dp0"
echo Setting up Stock Scanner Bot...
python -m venv .venv
call .venv\Scripts\activate.bat
pip install -r requirements.txt
echo.
echo Setup complete. You can now run run_scan_bot.bat
pause
