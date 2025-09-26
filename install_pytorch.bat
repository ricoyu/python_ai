@echo off
echo 安装 PyTorch...
echo.

cd /d "D:\Learning\python_ai"
call venv\Scripts\activate.bat

echo 正在安装 PyTorch...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

echo.
echo 安装完成！
echo 按任意键退出...
pause >nul

