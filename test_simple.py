#!/usr/bin/env python3
# -*- coding: utf-8 -*-

print("=== 简单测试 ===")

try:
    print("1. 测试导入...")
    from langchain_core.language_models.llms import BaseLLM
    from langchain_core.messages import AIMessage
    from transformers import AutoTokenizer, AutoModel
    print("✅ 导入成功!")
    
    print("2. 测试模型类创建...")
    class SimpleChatGLM3(BaseLLM):
        def _llm_type(self) -> str:
            return "SimpleChatGLM3"
        
        def _call(self, prompt: str, **kwargs) -> str:
            return f"测试响应: {prompt}"
        
        def _generate(self, prompts, **kwargs):
            from langchain_core.outputs import Generation, LLMResult
            generations = [[Generation(text=f"测试响应: {p}")] for p in prompts]
            return LLMResult(generations=generations)
    
    model = SimpleChatGLM3()
    print("✅ 模型类创建成功!")
    
    print("3. 测试调用...")
    result = model._call("你好")
    print(f"✅ 调用成功: {result}")
    
except Exception as e:
    print(f"❌ 错误: {e}")
    import traceback
    traceback.print_exc()

print("=== 测试完成 ===")
