#!/usr/bin/env python3
# -*- coding: utf-8 -*-

print("=== 测试修复后的 ChatGLM3 ===")

try:
    from transformers import AutoTokenizer, AutoModel, AutoConfig
    import torch
    
    model_path = 'E:\\HuggingFace\\chatglm3-6b'
    print(f"尝试加载模型: {model_path}")
    
    # 加载配置并修复
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    if not hasattr(config, 'num_hidden_layers'):
        if hasattr(config, 'num_layers'):
            config.num_hidden_layers = config.num_layers
        else:
            config.num_hidden_layers = 28
    
    # 加载分词器和模型
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=True)
    model = AutoModel.from_pretrained(
        model_path, 
        trust_remote_code=True, 
        device_map="auto",
        config=config
    )
    model = model.eval()
    
    print("✅ 模型加载成功!")
    
    # 测试推理 - 使用空历史记录
    print("测试推理...")
    response, history = model.chat(
        tokenizer,
        query="你好",
        history=[],  # 使用空列表而不是 None
        do_sample=True,
        max_length=100,
        temperature=0.7
    )
    print(f"✅ 推理成功: {response}")
    
except Exception as e:
    print(f"❌ 错误: {e}")
    import traceback
    traceback.print_exc()

print("=== 测试完成 ===")
