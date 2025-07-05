# Please install OpenAI SDK first: `pip3 install openai`
from Conversation import Conversation

file_path = "./test.txt"


conversation = Conversation(history_save_path=file_path, stream=True)

while True:
    text = input("用户：")

    response = conversation.send(text)