import json
from typing import Optional

from chatapi.Prompt import Prompt
from chatapi.MessageHistory import MessageHistory
from chatapi.ConnectionHandler import ConnectionHandler
from chatapi.ConnectionParams import ConnectionParams

class Conversation:
    history: MessageHistory
    stream: bool
    silence_mode: bool
    last_reply: str = ""
    last_reasoning_content: str | None = None
    last_logprobs: list | None = None

    def __init__(self, connection_handler: ConnectionHandler,
                 stream: bool = False):

        self.connection_handler = connection_handler

        self.history = MessageHistory(connection_handler.model_params)
        self.model_hp = self.history.model_hp

        self.stream = stream
        # 每个 Conversation 自己持有 response_format，不再写回共享的 connection_handler，
        # 这样同一个 handler 被多个线程/Conversation 并发复用时不会互相覆盖对方的设置。
        self.response_format = {"type": "text"}

    @classmethod
    def from_history(cls, history: MessageHistory, connection:ConnectionParams):
        connection = ConnectionHandler(history.model_hp, connection)
        item = cls(connection)
        item.history = history

        return item

    def init_session(self, prompt: Optional[Prompt] = None, session_title="New Session", **kwargs):
        self.history.set_session_title(session_title)
        self.history.set_language(language=prompt.language)

        if prompt is not None:
            self.history.set_system_prompt(prompt.system_message(), prompt_name=prompt.name)
            self.set_response_format(prompt.response_format)

            user_msg = prompt.user_message(**kwargs)
            if user_msg is not None:
                self.send(user_msg)

    def send(self, message, output_prefix="Assistant：", stream:Optional[bool]=None):
        self.history.user_message(message)

        result = self.connection_handler.send(self.history, output_prefix,
                                               stream=stream if stream is not None else self.stream,
                                               response_format=self.response_format)
        self.last_reply = result.reply
        self.last_reasoning_content = result.reasoning_content
        self.last_logprobs = result.logprobs

        self.history.assistant_message(self.last_reply, reasoning_content=self.last_reasoning_content,
                                        logprobs=self.last_logprobs)

        return self.last_reply

    def conversation_history(self):
        return self.history.session_content

    def set_response_format(self, response_format: dict):
        self.response_format = response_format

    def save_to(self, file_path: str):
        with open(file_path, "w") as f:
            f.write(json.dumps(self.conversation_history(), ensure_ascii=False) + "\n")

    def save_to_jsonl(self, jsonl_path: str):
        if jsonl_path.endswith(".jsonl"):
            with open(jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(self.conversation_history(), ensure_ascii=False) + "\n")
        else:
            raise "Not Jsonl file"
