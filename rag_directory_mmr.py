import os

from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers.string import StrOutputParser
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.runnables.base import RunnableParallel
from langchain_core.runnables.passthrough import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

loader = DirectoryLoader("D:\\Dropbox\\doc\杠精 怼人 口才", glob="*.md", show_progress=True)
docs = loader.load()

print(len(docs))

embeddings = HuggingFaceEmbeddings(model_name="google-bert/bert-base-chinese")

vector_store = FAISS.from_documents(docs, embeddings)

output = vector_store.similarity_search("中年人失业了怎么办?")
print(output)

retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 1} #获取最相似的1个
)

api_key = os.getenv("ZZZ_API_KEY")
base_url = os.getenv("ZZZ_API_BASE")

model = ChatOpenAI(api_key=api_key, base_url=base_url, )

template = """
值根据以下文档内容回答问题
{context}

问题: {question}"""

prompt = ChatPromptTemplate.from_template(template)

output_parser = StrOutputParser()

setup_and_retrieval = RunnableParallel({"context": retriever, "question": RunnablePassthrough()})

chain = setup_and_retrieval | prompt | model | output_parser
output = chain.invoke("中年失业怎么办?")
print(output)