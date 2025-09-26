# LangChain Local Model Demo 1 - ChatGLM3

这是一个使用 LangChain 框架集成 ChatGLM3 本地模型的示例代码。

## 修复的问题

1. **更新了过时的导入**：
   - 将 `langchain.llms.base.LLM` 更新为 `langchain_core.language_models.llms.BaseLLM`
   - 更新了相关的消息导入

2. **实现了缺失的抽象方法**：
   - 添加了 `_generate` 方法以满足 BaseLLM 接口要求

3. **添加了完整的类型注解**：
   - 为所有方法和属性添加了适当的类型提示

4. **改进了错误处理**：
   - 添加了模型加载检查
   - 添加了异常处理机制

## 安装依赖

```bash
# 激活虚拟环境
.\venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

## 使用方法

1. **基本使用**：
```python
from LangChainLocalModelDemo1 import ChatGLM3

model = ChatGLM3()
model.load_model("path/to/chatglm3-6b")
result = model.invoke("你的问题")
print(result.content)
```

2. **流式响应**：
```python
for chunk in model.stream("你的问题"):
    print(chunk, end="")
```

## 注意事项

- 需要下载 ChatGLM3 模型到本地
- 确保有足够的 GPU 内存（如果使用 GPU）
- 模型路径需要指向包含模型文件的目录

## 模型下载

可以从 Hugging Face 下载 ChatGLM3 模型：
```bash
git clone https://huggingface.co/THUDM/chatglm3-6b
```

## 运行测试

```bash
python LangChainLocalModelDemo1.py
```
