#!/usr/bin/env python3
# -*- coding: utf-8 -*-

print("=== 调试 ChatGLM3 模型加载 ===")

try:
    from transformers import AutoTokenizer, AutoModel, AutoConfig
    import torch
    
    model_path = 'E:\\HuggingFace\\chatglm3-6b'
    print(f"尝试加载模型: {model_path}")
    
    # 1. 加载配置
    print("1. 加载配置...")
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    print(f"配置类型: {type(config)}")
    print(f"配置属性: {dir(config)}")
    
    # 检查关键属性
    print(f"num_hidden_layers: {hasattr(config, 'num_hidden_layers')}")
    print(f"num_layers: {hasattr(config, 'num_layers')}")
    print(f"n_layer: {hasattr(config, 'n_layer')}")
    print(f"hidden_size: {hasattr(config, 'hidden_size')}")
    print(f"d_model: {hasattr(config, 'd_model')}")
    print(f"n_embd: {hasattr(config, 'n_embd')}")
    
    # 2. 修复配置
    print("2. 修复配置...")
    if not hasattr(config, 'num_hidden_layers'):
        if hasattr(config, 'num_layers'):
            config.num_hidden_layers = config.num_layers
            print(f"设置 num_hidden_layers = {config.num_layers}")
        elif hasattr(config, 'n_layer'):
            config.num_hidden_layers = config.n_layer
            print(f"设置 num_hidden_layers = {config.n_layer}")
        else:
            config.num_hidden_layers = 28
            print(f"设置 num_hidden_layers = 28 (默认值)")
    
    if not hasattr(config, 'hidden_size'):
        if hasattr(config, 'd_model'):
            config.hidden_size = config.d_model
            print(f"设置 hidden_size = {config.d_model}")
        elif hasattr(config, 'n_embd'):
            config.hidden_size = config.n_embd
            print(f"设置 hidden_size = {config.n_embd}")
        else:
            config.hidden_size = 4096
            print(f"设置 hidden_size = 4096 (默认值)")
    
    # 3. 加载分词器
    print("3. 加载分词器...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=True)
    print("✅ 分词器加载成功")
    
    # 4. 加载模型
    print("4. 加载模型...")
    model = AutoModel.from_pretrained(
        model_path, 
        trust_remote_code=True, 
        device_map="auto",
        config=config
    )
    print("✅ 模型加载成功")
    
    # 5. 测试推理
    print("5. 测试推理...")
    model = model.eval()
    
    response, history = model.chat(
        tokenizer,
        query="你好",
        history=[],
        do_sample=True,
        max_length=100,
        temperature=0.7
    )
    print(f"✅ 推理成功: {response}")
    
except Exception as e:
    print(f"❌ 错误: {e}")
    import traceback
    traceback.print_exc()

print("=== 调试完成 ===")
