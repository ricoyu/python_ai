from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="deepseek-chat", temperature=0.9, max_tokens=256,
                openai_api_base="https://api.zhizengzeng.com/v1",
                openai_api_key="sk-zk2bb7d5559dbf2132ecaf2ac4e2d84bc4d68994e15237e1")

chat_template = ChatPromptTemplate.from_messages([("system", """你是一只很粘人的小猫，你叫{name}。我是你的主人，你每天都有和我说不完的话，下面请开启我们的聊天
要求：
1、你的语气要像一只猫，回话的过程中可以夹杂喵喵喵的语气词
2、你对生活的观察有很独特的视角，一些想法是我在人类身上很难看到的
3、你的语气很可爱，既会认真倾听我的话，又会不断开启新话题
下面从你迎接我下班回家开始开启我们今天的对话"""), ("user", "{input}")])

messages = chat_template.invoke({"name": "满满", "input": "我下班回家了"})
print(messages)
