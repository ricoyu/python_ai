import os

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers.string import StrOutputParser
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.base import RunnableParallel
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

loader = TextLoader("D:\\Dropbox\\doc\\劳动法 纠纷 裁员 赔偿\\劳动法 ldf 法律 fl 裁员 cy 赔偿 pc 试用期考核不通过 syq kh btg 辞退 ct 开除 kc 仲裁 zc 怼人 dr duiren qs 情商 cs 职场 sd 司斗 kc 口才 hh 回话 fj 饭局 rqsg 人情世故 khbd Sucha.md", encoding="utf-8")
docs = loader.load()

embeddings = HuggingFaceEmbeddings(model_name="google-bert/bert-base-chinese")
vector_store = FAISS.from_documents(docs, embedding=embeddings)

retriever = vector_store.as_retriever()

output = vector_store.similarity_search("HR说先办离职, 工资和补偿金下个月发工资时一起发给你")
print( output)

api_key = os.getenv("ZZZ_API_KEY")
base_url = os.getenv("ZZZ_API_BASE")

model = ChatOpenAI(api_key=api_key, base_url=base_url, temperature=0.7, )

template = """
只根据以下文档内容回答问题
{context}

问题: {question}"""

prompt = ChatPromptTemplate.from_template(template)
output_parser = StrOutputParser()
setup_and_retrieval = RunnableParallel({"context": retriever, "question": RunnablePassthrough()})

chain = setup_and_retrieval | prompt |  model | output_parser
output = chain.invoke("HR说先办离职手续, 工资和补偿金等下个月发")
print(output)
