#!/usr/bin/env python3
# -*- coding: utf-8 -*-

print("=== ChatGLM3 LangChain 集成测试（不加载模型）===")
print()

# 检查依赖
def check_dependencies():
    missing_deps = []
    
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__} 已安装")
    except ImportError:
        missing_deps.append("torch")
        print("❌ PyTorch 未安装")
    
    try:
        import transformers
        print(f"✅ Transformers {transformers.__version__} 已安装")
    except ImportError:
        missing_deps.append("transformers")
        print("❌ Transformers 未安装")
    
    try:
        import sentencepiece
        print(f"✅ SentencePiece {sentencepiece.__version__} 已安装")
    except ImportError:
        missing_deps.append("sentencepiece")
        print("❌ SentencePiece 未安装")
    
    try:
        import accelerate
        print(f"✅ Accelerate {accelerate.__version__} 已安装")
    except ImportError:
        missing_deps.append("accelerate")
        print("❌ Accelerate 未安装")
    
    return missing_deps

print("检查依赖包...")
missing_deps = check_dependencies()
print()

if missing_deps:
    print("❌ 缺少以下依赖包:")
    for dep in missing_deps:
        print(f"   - {dep}")
    print()
    print("请运行以下命令安装:")
    print(f"   pip install {' '.join(missing_deps)}")
    print()
else:
    print("✅ 所有依赖包已安装!")
    print()

# 测试模型类创建
try:
    from langchain_core.language_models.llms import BaseLLM
    from langchain_core.messages import AIMessage
    from transformers import AutoTokenizer, AutoModel, AutoConfig
    from typing import Any, Dict, Iterator, List, Optional, Union
    
    class ChatGLM3(BaseLLM):
        max_token: int = 8192
        do_sample: bool = True
        temperature: float = 0.3
        top_p: float = 0.0
        tokenizer: Optional[Any] = None
        model: Optional[Any] = None
        history: List[Any] = []

        def __init__(self):
            super().__init__()

        @property
        def _llm_type(self) -> str:
            return "ChatGLM3"

        def _generate(self, prompts: List[str], stop: Optional[List[str]] = None, run_manager: Optional[Any] = None, **kwargs: Any) -> Any:
            """Generate responses for a list of prompts."""
            if self.model is None or self.tokenizer is None:
                raise RuntimeError("Model not loaded. Call load_model() first.")
            
            try:
                responses = []
                for prompt in prompts:
                    response, history = self.model.chat(
                        self.tokenizer,
                        query=prompt,
                        history=self.history,
                        do_sample=self.do_sample,
                        max_length=self.max_token,
                        temperature=self.temperature
                    )
                    self.history = history
                    responses.append(response)
                
                # Return in the format expected by LangChain
                from langchain_core.outputs import Generation, LLMResult
                generations = [[Generation(text=resp)] for resp in responses]
                return LLMResult(generations=generations)
            except Exception as e:
                raise RuntimeError(f"Error during model inference: {str(e)}")

        def _call(self, prompt: str, stop: Optional[List[str]] = None, run_manager: Optional[Any] = None, **kwargs: Any) -> str:
            """Call the model with a prompt and return the response."""
            if self.model is None or self.tokenizer is None:
                raise RuntimeError("Model not loaded. Call load_model() first.")
            
            try:
                response, history = self.model.chat(
                    self.tokenizer,
                    query=prompt,
                    history=self.history,
                    do_sample=self.do_sample,
                    max_length=self.max_token,
                    temperature=self.temperature
                )
                self.history = history
                return response
            except Exception as e:
                raise RuntimeError(f"Error during model inference: {str(e)}")

        def invoke(self, prompt: Union[str, Any], config: Optional[Dict[str, Any]] = None, **kwargs: Any) -> AIMessage:
            """Invoke the model and return an AIMessage."""
            if not isinstance(prompt, str):
                prompt = str(prompt)
            
            response = self._call(prompt, **kwargs)
            return AIMessage(content=response)

    model = ChatGLM3()
    print("✅ ChatGLM3 模型类创建成功!")
    print()
    
    # 不尝试加载实际模型，只测试类结构
    print("✅ 代码结构测试通过!")
    print()
    print("注意: 要使用实际模型，需要:")
    print("1. 下载 ChatGLM3 模型到 E:\\HuggingFace\\chatglm3-6b")
    print("2. 确保有足够的 GPU 内存")
    print("3. 运行完整的 LangChainLocalModelDemo1.py")
    
except Exception as e:
    print(f"❌ 创建模型类时出错: {e}")
    import traceback
    traceback.print_exc()

print()
print("=== 测试完成 ===")
