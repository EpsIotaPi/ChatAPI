import os, re
from typing import Union

from openai import OpenAI
from prompts.PromptLibrary import Prompt


class MessageHistory:
    __message_history = []
    prompt: str
    save_path: str

    def __init__(self, prompt:Union[str, Prompt], save_path=None):
        self.first_call = True

        if save_path is None:
            idx = 1
            possible_path = os.path.join("./history", f"conversation_{idx}.txt")
            while os.path.exists(possible_path):
                idx += 1
                possible_path = os.path.join("./history", f"conversation_{idx}.txt")
            save_path = possible_path

        if os.path.exists(save_path):
            self.load_history(save_path)
        else:
            self.save_path = save_path
            self.prompt = prompt

            self.__message_history = []

    def get_messages(self) -> list:
        return self.__message_history

    def user_message(self, message: str):
        if self.first_call:
            self.first_call = False
            if isinstance(self.prompt, Prompt):
                pass
            self.__save_content(role="system", content=self.prompt)

        self.__save_content(role="user", content=message)

    def assistant_message(self, message: str):
        self.__save_content(role="assistant", content=message)

    def __save_content(self, role: str, content: str):
        self.__message_history.append({"role": role, "content": content})

        with open(self.save_path, "a") as f:
            f.write("========== @{} ==========\n".format(role))
            f.write(content + "\n")

    def load_history(self, path):
        self.__message_history = []
        self.save_path = path
        self.prompt = ""
        self.first_call = False

        with open(self.save_path, "r") as f:
            role = None
            while True:
                line = f.readline()
                if not line:
                    break
                match = re.search(r"^={10}\s@([^\s=]+)\s={10}$", line)
                if match:
                    if role is not None:
                        self.__message_history.append({"role": role, "content": content})

                    if role == "system":
                        self.prompt = content

                    content = ""
                    role = match.group(1)
                else:
                    content += line
                self.__message_history.append({"role": role, "content": content})


class Conversation:
    history: MessageHistory

    def __init__(self, model="deepseek-chat", stream=False,
                 prompt: Union[str, Prompt] = "You are a helpful assistant", history_save_path=None):
        self.model = model
        self.stream = stream

        self.history = MessageHistory(prompt, history_save_path)

        deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
        base_url = "https://api.deepseek.com"
        self.client = OpenAI(api_key=deepseek_api_key, base_url=base_url)

    def send(self, message, output_prefix="助手："):
        self.history.user_message(message)
        response = self.client.chat.completions.create(
            messages=self.history.get_messages(),
            model=self.model,
            stream=self.stream
        )

        self.__response_handler(response, output_prefix)

    def __response_handler(self, response, output_prefix="助手："):
        print(output_prefix, end="")
        if self.stream:
            full_reply = ""
            for chunk in response:
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    print(delta.content, end="")
                    full_reply += delta.content
            print("")

        else:
            full_reply = response.choices[0].message.content
            print(full_reply)

        self.history.assistant_message(full_reply)
