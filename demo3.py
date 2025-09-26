from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
from langchain_openai import OpenAIEmbeddings

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
# 相似度的查询: 返回相似的分数, 分数越低, 相似度越高
resp = vector_store.similarity_search_with_score("咖啡猫", k=2)
print( resp)

retriver = RunnableLambda(vector_store.similarity_search).bind(k=1)
resp2 = retriver.batch(["咖啡猫", "鲨鱼"])
print( resp2)
