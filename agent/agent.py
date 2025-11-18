import json
from langchain.chat_models import init_chat_model

with open("../api_key.json") as f: json_key = json.load(f)
key = {}
for k in ['cstcloud', 'deepseek']:
    for m in json_key[k]['models']:
        key[m] = {"base_url": json_key[k]['domain'], "api_key": json_key[k]['key']}

name = "qwen3:235b"
model = init_chat_model(
    model=name,
    model_provider="openai",
    base_url="https://uni-api.cstcloud.cn/v1/",
    api_key=key[name],
)

question = "你好，请问你是"
result = model.invoke(question) #将question问题传递给model组件, 同步调用大模型生成结果
print(result.response_metadata)