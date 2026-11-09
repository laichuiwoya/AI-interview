@echo off
chcp 65001 >nul
cd /d "%~dp0"
set "PY=python"
if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"
echo [1/2] 正在检查依赖 pypdf / python-docx ...
"%PY%" -m pip install -q pypdf python-docx
echo [2/2] 正在启动服务...
start "" http://127.0.0.1:8765
"%PY%" server.py
pause
