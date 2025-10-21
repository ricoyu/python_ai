from langchain_core.prompts import ChatMessagePromptTemplate

prompt = "愿{subject}与你同在"

chat_message_prompt = ChatMessagePromptTemplate.from_template(role="Jedi", template=prompt)

message = chat_message_prompt.format(subject="force")
print( message) #ChatMessage类型

