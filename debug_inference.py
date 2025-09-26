#!/usr/bin/env python3
# -*- coding: utf-8 -*-

print("=== 调试模型推理 ===")

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
    
    # 测试不同的推理方法
    prompt = "你好"
    print(f"测试提示: {prompt}")
    
    # 方法1: 尝试 chat 方法
    print("\n1. 尝试 chat 方法...")
    try:
        response, history = model.chat(
            tokenizer,
            query=prompt,
            history=[],
            max_length=100,
            temperature=0.7,
            do_sample=True
        )
        print(f"✅ Chat 方法成功: {response}")
    except Exception as e:
        print(f"❌ Chat 方法失败: {e}")
        
        # 方法2: 尝试 generate 方法
        print("\n2. 尝试 generate 方法...")
        try:
            inputs = tokenizer(prompt, return_tensors="pt")
            print(f"输入形状: {inputs['input_ids'].shape}")
            
            outputs = model.generate(
                inputs['input_ids'],
                max_length=inputs['input_ids'].shape[1] + 50,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
            print(f"输出形状: {outputs.shape}")
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"原始响应: {response}")
            
            if prompt in response:
                response = response.replace(prompt, "").strip()
            print(f"✅ Generate 方法成功: {response}")
            
        except Exception as e2:
            print(f"❌ Generate 方法失败: {e2}")
            
            # 方法3: 尝试 forward 方法
            print("\n3. 尝试 forward 方法...")
            try:
                inputs = tokenizer(prompt, return_tensors="pt")
                print(f"输入 tokens: {inputs['input_ids']}")
                
                with torch.no_grad():
                    outputs = model(**inputs)
                    print(f"输出 logits 形状: {outputs.logits.shape}")
                    
                    # 获取最后一个 token 的 logits
                    logits = outputs.logits[0, -1, :]
                    print(f"最后一个 token 的 logits 形状: {logits.shape}")
                    
                    # 选择概率最高的 token
                    next_token_id = torch.argmax(logits).item()
                    print(f"下一个 token ID: {next_token_id}")
                    
                    response = tokenizer.decode([next_token_id], skip_special_tokens=True)
                    print(f"✅ Forward 方法成功: {response}")
                    
            except Exception as e3:
                print(f"❌ Forward 方法失败: {e3}")
    
except Exception as e:
    print(f"❌ 整体错误: {e}")
    import traceback
    traceback.print_exc()

print("\n=== 调试完成 ===")
