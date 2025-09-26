from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI

# 关键参数修改：指定DeepSeek的API地址和密钥
model = ChatOpenAI(
    model_name="gpt-4-turbo",  # 替换为具体的DeepSeek模型名称
    openai_api_base="https://api.zhizengzeng.com/v1",  # 智增增的DE BASE_URL
    # openai_api_key="sk-zk21423fce4ce26232550b939c668a7732b6895db76b7859",  # 智增增的KEY
    openai_api_key="sk-zk2bb7d5559dbf2132ecaf2ac4e2d84bc4d68994e15237e1",  # 智增增的KEY
    temperature=0.7,  # 可根据需要调整
    max_tokens=1024  # 可根据需要调整
)

# chatPromptTempalte = ChatPromptTemplate.from_messages([
#     SystemMessage(content="You are a helpful assistant."),
#     HumanMessage(content="What is the meaning of life?")
# ])
chatPromptTempalte = ChatPromptTemplate.from_messages([
    ("system", "你是一个乐于助人的助手, 用{language}尽你所能回答所有问题"),
    ("placeholder", "{chat_history}"),  # 用于插入历史消息
    ("human", "{input}")  # 用于接收当前用户输入
])

chain = chatPromptTempalte | model

# 报错聊天的历史记录
store = {}  # 所有用户的聊天记录都保存到store里面, 使用用户的sessionid作为key


def get_session_history(session_id):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


do_message = RunnableWithMessageHistory(
    chain,
    get_session_history,  # 回调函数
    # input_messages_key="my_msg"  # 每次聊天时候发送消息的KEY
    input_messages_key="input",  # 每次聊天时候发送消息的KEY
    history_messages_key="chat_history"  # 对应提示模板中的历史消息键
)

config = {"configurable": {"session_id": "sanshaoye666"}} # 给当前会话定义一个session_id

# 第一轮聊天
response = do_message.invoke(
    {
        "input": "你好, 我是三少爷",
        "language": "中文"
    },
    config=config
)
print(response.content)

# 第二轮聊天
response = do_message.invoke(
    {
        "input": "我叫什么名字?",
        "language": "中文"
    },
    config=config
)

print(response.content)

# 第三轮聊天, 流式输出
for resp in do_message.stream(
    {
        "input": "请给我讲一个笑话",
        "language": "中文"
    },
    config=config
):
    # 每一次resp都是一个token
    print(resp.content, end="")

