import json
from typing import Optional

from openai import OpenAI

from chat.Prompt import Prompt
from chat.HyperParams import ModelHyperParams
from chat.Connection import ConnectionParams
from chat.MessageHistory import MessageHistory


class Conversation:
    history: MessageHistory
    stream: bool
    silence_mode: bool
    last_reply: str = ""

    def __init__(self, hp: ModelHyperParams, connection: ConnectionParams,
                 stream=False, silence=False,
                 json_object: bool = False):

        self.history = MessageHistory(hp)
        self.model_hp = self.history.model_hp

        self.stream = stream
        self.silence_mode = silence
        self.response_format = {"type": "text"}
        if json_object:
            self.response_format = {"type": "json"}

        self.client = OpenAI(api_key=connection.api_key, base_url=connection.base_url)

    @classmethod
    def from_history(cls, history: MessageHistory, connection:ConnectionParams):
        item = cls(history.model_hp, connection)
        item.history = history

        return item

    @classmethod
    def from_hparams(cls, model:str, connection: ConnectionParams,
                     temperature = None, top_p = None, random_seed = None, **kwargs):

        hp = ModelHyperParams(model=model, temperature=temperature, top_p=top_p, random_seed=random_seed)
        return cls(hp, connection, **kwargs)

    def init_session(self, prompt: Optional[Prompt] = None, session_title="New Session", **kwargs):
        self.history.init_session(session_title, language=prompt.language)

        if prompt is not None:
            self.history.system_prompt(prompt.system_message(), prompt_name=prompt.name)

            if prompt.json_object:
                self.response_format = {"type": "json_object"}

            user_msg = prompt.user_message(**kwargs)
            if user_msg is not None:
                self.send(user_msg)

    def send(self, message, output_prefix="Assistant："):
        self.history.user_message(message)
        response = self.client.chat.completions.create(
            messages=self.history.messages(),
            model=self.model_hp.model,
            temperature=self.model_hp.temperature,
            top_p=self.model_hp.top_p,
            seed=self.model_hp.random_seed,
            stream=self.stream,
            response_format=self.response_format
        )

        return self.__response_handler(response, output_prefix)

    def __response_handler(self, response, output_prefix="Assistant："):
        self._print(output_prefix, end="")
        if self.stream:
            full_reply = ""
            for chunk in response:
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    print(delta.content, end="")
                    full_reply += delta.content
            self._print("")

        else:
            full_reply = response.choices[0].message.content
            self._print(full_reply)

        self.history.assistant_message(full_reply)
        self.last_reply = full_reply
        return full_reply

    def _print(self, obj, end: str | None = "\n",):
        if not self.silence_mode:
            print(obj, end=end)

    def conversation_history(self):
        return self.history.session_content()

    def save_to(self, file_path: str):
        with open(file_path, "w") as f:
            f.write(json.dumps(self.conversation_history(), ensure_ascii=False) + "\n")

    def save_to_jsonl(self, jsonl_path: str):
        if jsonl_path.endswith(".jsonl"):
            with open(jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(self.conversation_history(), ensure_ascii=False) + "\n")
        else:
            raise "Not Jsonl file"
