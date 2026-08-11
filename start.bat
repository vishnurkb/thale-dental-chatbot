@echo off
setlocal
cd /d "%~dp0"

set PYTHON_EXE=D:\v\Scripts\python.exe
if not exist "%PYTHON_EXE%" (
    echo [start] D:\v venv not found, falling back to "python" on PATH.
    set PYTHON_EXE=python
)

echo [start] Launching Thale Dental chatbot...
"%PYTHON_EXE%" "%~dp0run.py"

pause
