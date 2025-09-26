#!/usr/bin/env python3
# -*- coding: utf-8 -*-

print("=== 开始测试 ===")

# 测试依赖检查
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
    
    return missing_deps

print("检查依赖包...")
missing_deps = check_dependencies()
print()

if missing_deps:
    print("❌ 缺少以下依赖包:")
    for dep in missing_deps:
        print(f"   - {dep}")
else:
    print("✅ 所有依赖包已安装!")
    print()

print("=== 测试完成 ===")
