from fastapi import FastAPI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langserve import add_routes

# 关键参数修改：指定DeepSeek的API地址和密钥
model = ChatOpenAI(
    model_name="gpt-3.5-turbo",  # 替换为具体的DeepSeek模型名称
    openai_api_base="https://api.zhizengzeng.com/v1",  # 智增增的DE BASE_URL
    openai_api_key="sk-zk2bb7d5559dbf2132ecaf2ac4e2d84bc4d68994e15237e1",  # 智增增的KEY
    temperature=0.7,  # 可根据需要调整
    max_tokens=1024   # 可根据需要调整
)

parser = StrOutputParser()

prompt_template = ChatPromptTemplate.from_messages([
    ("system", "请将下面的内容翻译成{language}"),
    ("user", "{text}")
])
chain = prompt_template | model | parser
print("--------------------------------")
print( chain.invoke({"language": "西班牙语", "text": "我是一个有着八块腹肌和两块硕大胸肌的猛男 。"}))

# 把我们的程序部署成服务
app = FastAPI(title="自动翻译服务", version="v1.0", description="我的第一个使用LangChain构建的AI应用")
add_routes(app,
           chain,
           path="/chainDemo")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)