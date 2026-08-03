from openai import OpenAI
from google import genai
from google.genai import types

from chatapi.MessageHistory import MessageHistory
from chatapi.HyperParams import ModelHyperParams
from chatapi.ConnectionParams import ConnectionParams


class ConnectionHandler:
    def __init__(self, model_params:ModelHyperParams, connection_params:ConnectionParams,
                 silence=False):
        self.model_params = model_params
        self.connection_params = connection_params

        self.silence = silence
        self.response_format = {"type": "text"}

    @classmethod
    def from_hparams(cls, model_alias:str, connection: ConnectionParams, **kwargs):
        hp = ModelHyperParams(model_alias=model_alias, **kwargs)

        return cls(hp, connection, **kwargs)

    def set_silence_mode(self):
        self.silence = True

    def set_response_format(self, response_format:dict):
        self.response_format = response_format

    def send(self, message_history:MessageHistory, output_prefix="Assistant：", stream=False):
        """
        把 message 与 hp 整理成 send_content，并向LLM请求 response，用 response_handler 处理成 full_reply
        :param message_history:
        :param message:
        :param output_prefix:
        :return:
        """
        send_content = {
            "messages": message_history.messages,
            "model": self.model_params.model_alias,
            "temperature": self.model_params.model_alias,
            "top_p": self.model_params.top_p,
            "seed": self.model_params.random_seed,
            "stream": stream,
            "response_format": self.response_format
        }

        response = "this is the response from LLM"

        if stream and self.silence:
            print("You set up silence mode, but stream response still will be printed.")

        full_reply = self._response_handler(response, output_prefix, stream)
        return full_reply

    def _response_handler(self, response, output_prefix, stream):
        """
        对收到的 LLM response 进行处理、打印。流式信息和普通信息需要分开处理。
        :param response:
        :param output_prefix:
        :return:
        """
        self._print(output_prefix, end="")
        full_reply = response
        self._print(full_reply)

        return full_reply

    def _print(self, obj, **kwargs):
        if not self.silence:
            print(obj, **kwargs)


class OpenAIConnectionHandler(ConnectionHandler):
    def __init__(self, model_params:ModelHyperParams, connection_params:ConnectionParams,
                 silence=False):
        super().__init__(model_params, connection_params, silence)

        self.client = OpenAI(api_key=connection_params.api_key,
                             base_url=connection_params.base_url)
        self.last_reasoning_content = None
        self.last_logprobs = None

    def send(self, message_history: MessageHistory, output_prefix="Assistant：", stream=False):
        msg_history = message_history.messages
        messages = [
            {k: m[k] for k in  ("role", "content", "reasoning_content") if k in m and m[k] is not None}
            for m in msg_history
        ]

        response = self.client.chat.completions.create(
            messages=messages,

            model=self.model_params.model_alias,
            temperature=self.model_params.temperature,
            top_p=self.model_params.top_p,
            frequency_penalty=self.model_params.frequency_penalty,
            presence_penalty=self.model_params.presence_penalty,

            seed=self.model_params.random_seed,
            max_completion_tokens = self.model_params.max_completion_tokens,
            reasoning_effort = self.model_params.reasoning_effort,

            logprobs=self.model_params.logprobs,
            top_logprobs=self.model_params.top_logprobs if self.model_params.logprobs else None,

            extra_body=self.model_params.extra_body,

            stream=stream,
            response_format=self.response_format
        )

        if stream and self.silence:
            print("You set up silence mode, but stream response still will be printed.")

        full_reply, self.last_reasoning_content, self.last_logprobs = self._response_handler(response, output_prefix, stream)

        return full_reply

    def _response_handler(self, response, output_prefix, stream):
        if stream:
            full_reply = ""
            reasoning_content = None
            logprobs_entries = []
            for chunk in response:
                if len(chunk.choices) != 0:
                    delta = chunk.choices[0].delta
                    chunk_logprobs = getattr(chunk.choices[0].logprobs, "content", None) if chunk.choices[0].logprobs else None
                    if chunk_logprobs:
                        logprobs_entries.extend(chunk_logprobs)
                    if not delta:
                        continue
                    reasoning_delta = getattr(delta, "reasoning_content", None)
                    if reasoning_delta:
                        if reasoning_content is None:
                            reasoning_content = ""
                            self._print("Reasoning: ", end="")
                        self._print(reasoning_delta, end="")
                        reasoning_content += reasoning_delta
                    elif delta.content:
                        if full_reply == "":
                            print("")
                            self._print(output_prefix, end="")
                        print(delta.content, end="")
                        full_reply += delta.content
            self._print("")
            logprobs = self._serialize_logprobs(logprobs_entries)
        else:
            message = response.choices[0].message
            full_reply = message.content or ""
            reasoning_content = getattr(message, "reasoning_content", None) or None
            choice_logprobs = getattr(response.choices[0].logprobs, "content", None) if response.choices[0].logprobs else None
            logprobs = self._serialize_logprobs(choice_logprobs)
            self._print(full_reply)

        return full_reply, reasoning_content, logprobs

    @staticmethod
    def _serialize_logprobs(entries):
        """
        把 openai SDK 返回的 ChatCompletionTokenLogprob 对象列表转换成可 JSON 序列化的 dict 列表，
        便于随消息一起落盘到 MessageHistory。
        """
        if not entries:
            return None
        return [
            {
                "token": entry.token,
                "logprob": entry.logprob,
                "top_logprobs": [
                    {"token": t.token, "logprob": t.logprob}
                    for t in entry.top_logprobs
                ] if getattr(entry, "top_logprobs", None) else None,
            }
            for entry in entries
        ]


class GoogleConnectionHandler(ConnectionHandler):
    # gemma 系列不支持 thinking，只对 gemini 系列模型自动开启 include_thoughts，
    # 避免给不支持的模型传 thinking_config 导致请求报错。
    _THINKING_CAPABLE_PREFIX = "gemini"

    def __init__(self, model_params:ModelHyperParams, connection_params:ConnectionParams,
                 silence=False, include_thoughts: bool = True):
        super().__init__(model_params, connection_params, silence)

        self.client = genai.Client(api_key=connection_params.api_key)
        self.last_reasoning_content = None

        thinking_config = None
        if include_thoughts and self.model_params.model_alias.startswith(self._THINKING_CAPABLE_PREFIX):
            thinking_config = types.ThinkingConfig(include_thoughts=True)

        self.config = types.GenerateContentConfig(temperature=self.model_params.temperature,
                                                  top_p=self.model_params.top_p,
                                                  seed=self.model_params.random_seed,
                                                  thinking_config=thinking_config)

        self.chat = self.client.chats.create(model=self.model_params.model_alias, config=self.config)

    def send(self, message_history:MessageHistory, output_prefix="Assistant：", stream=False):
        sys_prompt = message_history.sys_prompt
        if sys_prompt is not None and len(message_history) == 2: # 第一次提示的时候更新。此时有一条系统提示，和一条普通提示
            self.update_sys_instruction(sys_prompt)

        message = message_history.last_message

        if stream and self.silence:
            print("You set up silence mode, but stream response still will be printed.")

        if stream:
            response = self.chat.send_message_stream(message)
        else:
            response = self.chat.send_message(message)

        full_reply, self.last_reasoning_content = self._response_handler(response, output_prefix, stream)

        return full_reply

    def _response_handler(self, response, output_prefix, stream):
        self._print(output_prefix, end="")
        if stream:
            full_reply = ""
            reasoning_content = None
            printed_reasoning_prefix = False
            for chunk in response:
                candidates = getattr(chunk, "candidates", None) or []
                if not candidates:
                    continue
                reply_part, thought_part = self._split_parts(getattr(candidates[0].content, "parts", None))
                if thought_part:
                    if not printed_reasoning_prefix:
                        self._print("\nReasoning: ", end="")
                        printed_reasoning_prefix = True
                    self._print(thought_part, end="", flush=True)
                    reasoning_content = (reasoning_content or "") + thought_part
                if reply_part:
                    self._print(reply_part, end="", flush=True)
                    full_reply += reply_part
            self._print("")
        else:
            candidates = getattr(response, "candidates", None) or []
            parts = getattr(candidates[0].content, "parts", None) if candidates else None
            full_reply, reasoning_content = self._split_parts(parts)
            self._print(full_reply)

        return full_reply, reasoning_content

    def update_sys_instruction(self, sys_instruction:str):
        self.config.system_instruction = sys_instruction
        self.chat = self.client.chats.create(model=self.model_params.model_alias, config=self.config)

    @staticmethod
    def _split_parts(parts):
        """
        Gemini 把思考内容和正文分别放在 content.parts 里，用 part.thought 标记区分，
        而不是像 DeepSeek/本地模型那样用独立字段或内联标签。这里按标记拆成 (正文, 推理) 两段文本。
        """
        reply_text = ""
        thought_text = ""
        for part in (parts or []):
            text = getattr(part, "text", None)
            if not text:
                continue
            if getattr(part, "thought", False):
                thought_text += text
            else:
                reply_text += text
        return reply_text, (thought_text or None)





