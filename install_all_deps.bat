@echo off
echo ========================================
echo 安装所有缺失的依赖包
echo ========================================
echo.

cd /d "D:\Learning\python_ai"
call venv\Scripts\activate.bat

echo 1. 安装 PyTorch...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

echo.
echo 2. 安装其他依赖...
pip install transformers
pip install sentencepiece
pip install langchain-core
pip install accelerate

echo.
echo 3. 验证安装...
python -c "import torch; print('PyTorch version:', torch.__version__)"
python -c "import transformers; print('Transformers version:', transformers.__version__)"
python -c "import sentencepiece; print('SentencePiece version:', sentencepiece.__version__)"

echo.
echo ========================================
echo 安装完成！现在可以运行测试了
echo ========================================
echo.
echo 按任意键运行测试...
pause >nul

echo 运行测试...
python LangChainLocalModelDemo1.py

echo.
echo 按任意键退出...
pause >nul
