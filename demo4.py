# 提示模板
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

model = ChatOpenAI(
    model_name="gpt-4-turbo",  # 替换为具体的DeepSeek模型名称
    openai_api_base="https://api.zhizengzeng.com/v1",  # 智增增的DE BASE_URL
    # openai_api_key="sk-zk21423fce4ce26232550b939c668a7732b6895db76b7859",  # 智增增的KEY
    openai_api_key="sk-zk2bb7d5559dbf2132ecaf2ac4e2d84bc4d68994e15237e1",  # 智增增的KEY
    temperature=0.7,  # 可根据需要调整
    max_tokens=1024  # 可根据需要调整
)

documents = [
    Document(
        page_content="狗是伟大的伴侣, 以其忠诚和友好而闻名",
        metadata={"source": "哺乳动物宠物文档"},
    ),
    Document(
        page_content="猫是独立的宠物, 通常喜欢自己的空间",
        metadata={"source": "哺乳动物宠物文档"},
    ),
    Document(
        page_content="金鱼是初学者的流行宠物, 需要相对简单的护理",
        metadata={"source": "鱼类宠物文档"},
    ),
    Document(
        page_content="鹦鹉是聪明的鸟类, 能够模仿人类的语言",
        metadata={"source": "鸟类宠物文档"},
    ),
    Document(
        page_content="兔子是社交动物, 需要足够的空间跳跃",
        metadata={"source": "哺乳动物宠物文档"},
    )
]

# 实例化一个向量数据库
vector_store = Chroma.from_documents(documents, OpenAIEmbeddings(
    openai_api_base="https://api.zhizengzeng.com/v1",  # 智增增的API地址
    openai_api_key="sk-zk2bb7d5559dbf2132ecaf2ac4e2d84bc4d68994e15237e1"  # 你的智增增密钥
))

retriver = RunnableLambda(vector_store.similarity_search).bind(k=1)

message = """
使用提供的上下文仅回答这个问题
{question}
上下文:
{context}
"""

prompt_temp = ChatPromptTemplate.from_messages([("human", message)])

# RunnablePassthrough允许我们将用户的问题传递给prompt和model, 但是是稍后再传递的
chain = {"question": RunnablePassthrough(), "context": retriver} | prompt_temp | model

# 结合了上下文的回答就不是像百度搜索的结果那么笼统了, 而是结合上下文的回答
resp = chain.invoke("请介绍一下猫")
print( resp.content) # 输出 猫是一种独立的宠物，它们通常喜欢拥有自己的空间。
