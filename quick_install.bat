@echo off
echo 快速安装所有依赖包...
echo.

cd /d "D:\Learning\python_ai"
call venv\Scripts\activate.bat

echo 正在安装 PyTorch...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

echo 正在安装其他依赖...
pip install transformers
pip install sentencepiece
pip install langchain-core
pip install accelerate

echo 安装完成！
echo 按任意键退出...
pause >nul

