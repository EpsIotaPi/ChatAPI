import os
from chatapi.Conversation import Conversation
from chatapi.Prompt import PromptManager
from chatapi.ConnectionHandler import OpenAIConnectionHandler, GoogleConnectionHandler
from chatapi.ConnectionParams import OpenAIConnectionParams, DeepSeekConnectionParams, GoogleConnectionParams
from chatapi.HyperParams import ModelHyperParams


file_path = input("键入保存历史记录的路径（默认为 ./chatapi/history/default.json）：")

if file_path == "":
    file_path = "chatapi/history/default.json"
elif not os.path.isfile(file_path):
    file_path = "./chatapi/history/{}.json".format(file_path)

connection_params = DeepSeekConnectionParams()
connection_params.deepseek_pro()
hp = ModelHyperParams(model=connection_params.model_alias, temperature=0, random_seed=114514)
connection_handler = OpenAIConnectionHandler(hp, connection_params, silence=False, stream=True)

prompt = PromptManager().get_prompt("test", lang="zh")

conversation = Conversation(connection_handler)
conversation.init_session(prompt=prompt)

while True:
    text = input("用户：")
    if text == "/end":
        break
    response = conversation.send(text)

conversation.save_to(file_path)