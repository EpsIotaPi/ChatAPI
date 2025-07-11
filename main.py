# Please install OpenAI SDK first: `pip3 install openai`
from Conversation import Conversation
from prompts.PromptLibrary import Prompt

file_path = "./test.txt"

prompt = Prompt("basic")


conversation = Conversation(history_save_path=file_path, stream=True, prompt=prompt("eng"))

while True:
    text = input("用户：")

    response = conversation.send(text)