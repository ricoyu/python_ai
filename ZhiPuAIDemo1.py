import os

from zhipuai import ZhipuAI

"""
演示接入智谱AI
"""
api_key = os.getenv("ZHIPU_API_KEY")
print(api_key)

prompt = "中国的首都是哪里？"

model = ZhipuAI(api_key=api_key)
response = model.chat.completions.create(model="glm-4.5", messages=[{"role": "user", "content": prompt},
                                                                    {"role": "system", "content": "回答要简洁高效"}])
answer = response.choices[0].message.content
print(answer)

# response = model.chat.completions.create(model="glm-4.5", messages=[
#                                                                     {"role": "system", "content": "回答要简洁高效"},
#                                                                     {"role": "user", "content": prompt},
#                                                                     {"role": "assistant", "content": answer},
#                                                                     {"role": "user", "content": "这个城市有什么特色美食呢?"},
# ])
# print(response.choices[0].message.content)

# 流式输出
response = model.chat.completions.create(model="glm-4.5", messages=[
    {"role": "system", "content": "回答要简洁高效"},
    {"role": "user", "content": prompt},
    {"role": "assistant", "content": answer},
    {"role": "user", "content": "这个城市有什么特色美食呢?"}],
                                         stream=True)
for chunk in response:
    content = chunk.choices[0].delta.content
    print(content, end="")
