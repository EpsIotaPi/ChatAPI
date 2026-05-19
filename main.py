import os
from chat.Conversation import Conversation
from chat.Prompt import PromptManager
from chat.Connection import OpenAIConnection, DeepSeekConnection, GoogleConnection
from chat.HyperParams import ModelHyperParams


file_path = input("键入保存历史记录的路径（默认为 ./chat/history/default.json）：")

if file_path == "":
    file_path = "./chat/history/default.json"
elif not os.path.isfile(file_path):
    file_path = "./chat/history/{}.json".format(file_path)


connection = DeepSeekConnection()
hp = ModelHyperParams(model=connection.deepseek_flash, temperature=1.3, random_seed=114514)
prompt = PromptManager().get_prompt("test", lang="zh")


conversation = Conversation(hp, connection, silence=False)
conversation.init_session(prompt)

while True:
    text = input("用户：")
    if text == "/end":
        break
    response = conversation.send(text)

conversation.save_to(file_path)