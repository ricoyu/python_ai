"""
RAG的使用
"""
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.vectorstores import chroma
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import bs4
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai.types import VectorStore

model = ChatOpenAI(
    model_name="gpt-4-turbo",
    openai_api_base="https://api.zhizengzeng.com/v1",  # 智增增的DE BASE_URL
    openai_api_key="sk-zk2bb7d5559dbf2132ecaf2ac4e2d84bc4d68994e15237e1",  # 智增增的KEY
    temperature=0.7,  # 可根据需要调整
    max_tokens=1024  # 可根据需要调整
)

# 1. 加载数据: - 网上的一篇博客数据 https://lilianweng.github.io/posts/2023-06-23-agent/
loader= WebBaseLoader(web_paths=["https://lilianweng.github.io/posts/2023-06-23-agent/"])
bs_kwargs= dict(
    parse_only= bs4.SoupStrainer(class_=("post-header", "post-title", "post-content"))
    # parse_only= bs4.SoupStrainer(class_="post-title")
)

docs = loader.load()
# print(len(docs))
# print(docs)

#2 文本分割
"""
chunk_size表示每个片段包含多少字符
chunk_overlap 允许重复4个字符
"""
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=4)
# splits = splitter.split_text(docs[0].metadata["description"])
splits = splitter.split_documents(docs)

# for s in splits:
#     print(s, end="***\n\r")

# 3 存储
# vector_store = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings(), persist_directory="./chroma_db")
vector_store = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings(
    openai_api_base="https://api.zhizengzeng.com/v1",  # 智增增的API地址
    openai_api_key="sk-zk2bb7d5559dbf2132ecaf2ac4e2d84bc4d68994e15237e1"
))

# 4 检索器
retriever = vector_store.as_retriever()

system_prompt = """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, say that you don't know. 
Use three sentences maximum and keep the answer concise.\n
                    
{context}
"""

# 5 创建一个问题的模板, 提问和回答的历史记录模板
prompt = ChatPromptTemplate.from_messages( [
    ("system", system_prompt),
    # MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

#6 整合AI大模型
# chain = model.bind_retriever(retriever)

# 7 创建链
#第一个chain可以做问答
chain1 = create_stuff_documents_chain(model, prompt)
# 第二个chain在第一个chain的基础上增加了检索器
chain2 = create_retrieval_chain(retriever, chain1)

resp = chain2.invoke({"input": "What is Task Decomposition?"})
print( resp["answer"])