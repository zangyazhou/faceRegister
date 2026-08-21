@echo off
setlocal
REM 人脸识别签到系统 启动脚本
cd /d "%~dp0"
set "PYTHON_CMD=python"
if exist ".venv\Scripts\python.exe" set "PYTHON_CMD=.venv\Scripts\python.exe"
if exist "venv\Scripts\python.exe" set "PYTHON_CMD=venv\Scripts\python.exe"

%PYTHON_CMD% -c "import sys; assert (3, 10) <= sys.version_info[:2] <= (3, 12)" >nul 2>&1
if errorlevel 1 (
    echo 未找到 Python 3.10~3.12，请先按 README 创建虚拟环境。
    pause
    exit /b 1
)

echo 正在启动服务... 浏览器访问 http://127.0.0.1:5000
echo 按 Ctrl+C 停止
%PYTHON_CMD% -m flask --app app run
pause
