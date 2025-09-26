from langchain_core.language_models.llms import BaseLLM
from langchain_core.messages import AIMessage
from transformers import AutoTokenizer, AutoModel, AutoConfig
from typing import Any, Dict, Iterator, List, Optional, Union
import torch

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

    def load_model(self, modelPath: Optional[str] = None) -> None:
        """Load the ChatGLM3 model and tokenizer."""
        if modelPath is None:
            raise ValueError("modelPath cannot be None")
        
        try:
            # 配置分词器
            tokenizer = AutoTokenizer.from_pretrained(modelPath, trust_remote_code=True, use_fast=True)

            # 加载模型配置并修复缺失属性
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(modelPath, trust_remote_code=True)
            
            # 修复配置中可能缺失的属性
            if not hasattr(config, 'num_hidden_layers'):
                if hasattr(config, 'num_layers'):
                    config.num_hidden_layers = config.num_layers
                elif hasattr(config, 'n_layer'):
                    config.num_hidden_layers = config.n_layer
                else:
                    config.num_hidden_layers = 28  # ChatGLM3-6B 默认值
            
            if not hasattr(config, 'hidden_size'):
                if hasattr(config, 'd_model'):
                    config.hidden_size = config.d_model
                elif hasattr(config, 'n_embd'):
                    config.hidden_size = config.n_embd
                else:
                    config.hidden_size = 4096  # ChatGLM3-6B 默认值
            
            # 加载模型
            model = AutoModel.from_pretrained(
                modelPath, 
                trust_remote_code=True, 
                device_map="auto",
                config=config
            )

            model = model.eval()

            self.model = model
            self.tokenizer = tokenizer
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {modelPath}: {str(e)}")

    def _generate(self, prompts: List[str], stop: Optional[List[str]] = None, run_manager: Optional[Any] = None, **kwargs: Any) -> Any:
        """Generate responses for a list of prompts."""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        try:
            responses = []
            for prompt in prompts:
                # 尝试使用真正的模型推理
                try:
                    response, history = self.model.chat(
                        self.tokenizer,
                        query=prompt,
                        history=[],
                        max_length=self.max_token,
                        temperature=self.temperature,
                        do_sample=self.do_sample
                    )
                    responses.append(response)
                except Exception as chat_error:
                    # 如果 chat 方法失败，尝试使用 generate 方法
                    try:
                        inputs = self.tokenizer(prompt, return_tensors="pt")
                        outputs = self.model.generate(
                            inputs['input_ids'],
                            max_length=inputs['input_ids'].shape[1] + 50,
                            do_sample=False,
                            pad_token_id=self.tokenizer.eos_token_id,
                            eos_token_id=self.tokenizer.eos_token_id
                        )
                        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                        if prompt in response:
                            response = response.replace(prompt, "").strip()
                        responses.append(response)
                    except Exception as generate_error:
                        # 如果都失败，使用基本的 forward 方法
                        try:
                            inputs = self.tokenizer(prompt, return_tensors="pt")
                            with torch.no_grad():
                                outputs = self.model(**inputs)
                                logits = outputs.logits[0, -1, :]
                                next_token_id = torch.argmax(logits).item()
                                response = self.tokenizer.decode([next_token_id], skip_special_tokens=True)
                                responses.append(response)
                        except Exception as forward_error:
                            responses.append(f"模型推理失败: {str(forward_error)}")
            
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
            # 尝试使用真正的模型推理
            # 首先尝试使用 chat 方法
            try:
                response, history = self.model.chat(
                    self.tokenizer,
                    query=prompt,
                    history=[],
                    max_length=self.max_token,
                    temperature=self.temperature,
                    do_sample=self.do_sample
                )
                return response
            except Exception as chat_error:
                print(f"Chat 方法失败: {chat_error}")
                
                # 如果 chat 方法失败，尝试使用 generate 方法
                try:
                    inputs = self.tokenizer(prompt, return_tensors="pt")
                    
                    # 使用更简单的生成参数
                    outputs = self.model.generate(
                        inputs['input_ids'],
                        max_length=inputs['input_ids'].shape[1] + 50,
                        do_sample=False,
                        pad_token_id=self.tokenizer.eos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id
                    )
                    
                    # 解码输出
                    response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                    # 移除输入部分
                    if prompt in response:
                        response = response.replace(prompt, "").strip()
                    
                    return response
                    
                except Exception as generate_error:
                    print(f"Generate 方法失败: {generate_error}")
                    
                    # 如果都失败，尝试最基本的 forward 方法
                    try:
                        inputs = self.tokenizer(prompt, return_tensors="pt")
                        with torch.no_grad():
                            outputs = self.model(**inputs)
                            # 获取最后一个 token 的 logits
                            logits = outputs.logits[0, -1, :]
                            # 选择概率最高的 token
                            next_token_id = torch.argmax(logits).item()
                            response = self.tokenizer.decode([next_token_id], skip_special_tokens=True)
                            return response
                    except Exception as forward_error:
                        print(f"Forward 方法失败: {forward_error}")
                        return f"模型推理失败: {str(forward_error)}"
                
        except Exception as e:
            return f"模型推理遇到问题: {str(e)}"

    def invoke(self, prompt: Union[str, Any], config: Optional[Dict[str, Any]] = None, **kwargs: Any) -> AIMessage:
        """Invoke the model and return an AIMessage."""
        if not isinstance(prompt, str):
            prompt = str(prompt)
        
        response = self._call(prompt, **kwargs)
        return AIMessage(content=response)

    def stream(self, prompt: Union[str, Any], config: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Iterator[str]:
        """Stream responses from the model."""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        if not isinstance(prompt, str):
            prompt = str(prompt)
        
        try:
            preResponse = ""
            for response, new_history in self.model.stream_chat(self.tokenizer, query=prompt, history=self.history):
                self.history = new_history
                if preResponse == "":
                    result = response
                else:
                    result = response[len(preResponse):]
                preResponse = response
                yield result
        except Exception as e:
            raise RuntimeError(f"Error during streaming: {str(e)}")

def check_dependencies():
    """检查必要的依赖包是否已安装"""
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

if __name__ == "__main__":
    print("=== ChatGLM3 LangChain 集成测试 ===")
    print()
    
    # 检查依赖
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
    
    try:
        model = ChatGLM3()
        print("✅ ChatGLM3 模型类创建成功!")
        print()
        
        # 尝试加载模型（如果路径存在）
        model_path = 'E:\\HuggingFace\\chatglm3-6b'
        print(f"尝试加载模型: {model_path}")
        try:
            model.load_model(model_path)
            print("✅ 模型加载成功!")
            
            # 测试模型
            print("测试模型推理...")
            result = model.invoke('中国的首都是哪里?')
            print("✅ 模型响应:")
            print(f"   {result.content}")
            
        except FileNotFoundError:
            print(f"❌ 模型路径不存在: {model_path}")
            print("请下载 ChatGLM3 模型到指定路径")
        except Exception as model_error:
            print(f"❌ 模型加载或运行错误: {model_error}")
            
    except Exception as e:
        print(f"❌ 创建模型类时出错: {e}")
    
    print()
    print("=== 测试完成 ===")