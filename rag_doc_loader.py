from langchain_community.document_loaders import DirectoryLoader

loader = DirectoryLoader("D:\Dropbox\doc\劳动法 纠纷 裁员 赔偿", glob="*.md", show_progress=True)
doc = loader.load()
print(len(doc))
