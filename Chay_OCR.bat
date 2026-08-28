@echo off
cd /d "%~dp0"
if exist ".venv\Scripts\pythonw.exe" (
    start "" ".venv\Scripts\pythonw.exe" run_gui.py
) else (
    echo Khong tim thay moi truong ao .venv!
    pause
)
exit
