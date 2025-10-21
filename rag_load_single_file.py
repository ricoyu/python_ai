from langchain_community.document_loaders import TextLoader

loader = TextLoader("D:\\Dropbox\\doc\\杠精 怼人 口才\\杠精 gj 怼人dr 口才kc 辩论bl Sucha.md", encoding="utf-8")
doc = loader.load()
print(doc[0].page_content)