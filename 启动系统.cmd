@echo off
chcp 65001 >nul
cd /d "%~dp0"
set "GUI_PYTHON=D:\Anaconda3\envs\yolov10\python.exe"
if defined EYE_PYTHON set "GUI_PYTHON=%EYE_PYTHON%"
if not exist "%GUI_PYTHON%" (
 echo 未找到运行环境，请安装环境或设置 EYE_PYTHON。
 pause
 exit /b 1
)
"%GUI_PYTHON%" scripts\launch_desktop.py
if errorlevel 1 pause
