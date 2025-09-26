#!/usr/bin/env python3
# -*- coding: utf-8 -*-

print("=== 最终测试 ChatGLM3 ===")

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
    
    # 使用修复后的方法
    print("测试 generate 方法...")
    
    prompt = "你好"
    inputs = tokenizer(prompt, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=100,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # 解码输出
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # 移除输入部分，只保留生成的部分
    if prompt in response:
        response = response.replace(prompt, "").strip()
    
    print(f"✅ 生成成功: {response}")
    
except Exception as e:
    print(f"❌ 错误: {e}")
    import traceback
    traceback.print_exc()

print("=== 测试完成 ===")
