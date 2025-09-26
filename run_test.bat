@echo off
echo 运行 ChatGLM3 LangChain 集成测试...
echo.

cd /d "D:\Learning\python_ai"
call venv\Scripts\activate.bat
python LangChainLocalModelDemo1.py

echo.
echo 按任意键退出...
pause >nul
