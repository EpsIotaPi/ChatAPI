# Please install OpenAI SDK first: `pip3 install openai`
import os
from chat.Conversation import Conversation
from chat.Prompt import Prompt


file_path = input("键入保存历史记录的路径（默认为 ./history/default.txt）：")

if file_path == "":
    file_path = "chat/history/default.txt"
elif not os.path.isfile(file_path):
    file_path = "./history/{}.txt".format(file_path)


prompt = Prompt("basic")


conversation = Conversation(history_save_path=file_path, stream=True, prompt=prompt("eng"))

while True:
    text = input("用户：")

    response = conversation.send(text)