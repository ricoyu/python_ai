#!/usr/bin/env python3
# -*- coding: utf-8 -*-

try:
    import sentencepiece as spm
    print("✅ sentencepiece 导入成功!")
    print(f"版本: {spm.__version__}")
except ImportError as e:
    print(f"❌ sentencepiece 导入失败: {e}")

try:
    from transformers import AutoTokenizer
    print("✅ transformers 导入成功!")
except ImportError as e:
    print(f"❌ transformers 导入失败: {e}")

try:
    import torch
    print("✅ torch 导入成功!")
    print(f"版本: {torch.__version__}")
except ImportError as e:
    print(f"❌ torch 导入失败: {e}")

print("\n所有依赖检查完成!")
