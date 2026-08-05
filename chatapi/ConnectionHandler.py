import threading
import weakref
from typing import NamedTuple, Optional

from openai import OpenAI
from google import genai
from google.genai import types

from chatapi.MessageHistory import MessageHistory
from chatapi.HyperParams import ModelHyperParams
from chatapi.ConnectionParams import ConnectionParams


class SendResult(NamedTuple):
    """
    单次 send() 的结果。之前 reasoning_content/logprobs 是先写回 self.last_*
    再由调用方读回，多线程共享同一个 handler 实例时会被其他线程的请求覆盖；
    改成随返回值传递后，每次调用的结果只属于这次调用，天然线程安全。
    """
    reply: str
    reasoning_content: Optional[str] = None
    logprobs: Optional[list] = None


class ConnectionHandler:
    def __init__(self, model_params:ModelHyperParams, connection_params:ConnectionParams,
                 silence=False):
        self.model_params = model_params
        self.connection_params = connection_params

        self.silence = silence
        # 仅作为 send() 未显式传入 response_format 时的默认值。多个调用方共享同一个
        # handler 实例时不应再依赖 set_response_format() 修改这个默认值，而应该
        # 通过 send() 的 response_format 参数显式传入，避免并发请求互相覆盖。
        self.response_format = {"type": "text"}

    @classmethod
    def from_hparams(cls, model_alias:str, connection: ConnectionParams, **kwargs):
        hp = ModelHyperParams(model_alias=model_alias, **kwargs)

        return cls(hp, connection, **kwargs)

    def set_silence_mode(self):
        self.silence = True

    def set_response_format(self, response_format:dict):
        self.response_format = response_format

    def send(self, message_history:MessageHistory, output_prefix="Assistant：", stream=False,
              response_format:Optional[dict]=None) -> SendResult:
        """
        把 message 与 hp 整理成 send_content，并向LLM请求 response，用 response_handler 处理成 full_reply
        :param message_history:
        :param output_prefix:
        :param stream:
        :param response_format: 本次请求使用的 response_format，缺省时才回落到 self.response_format
        :return:
        """
        response_format = response_format if response_format is not None else self.response_format

        send_content = {
            "messages": message_history.messages,
            "model": self.model_params.model_alias,
            "temperature": self.model_params.model_alias,
            "top_p": self.model_params.top_p,
            "seed": self.model_params.random_seed,
            "stream": stream,
            "response_format": response_format
        }

        response = "this is the response from LLM"

        if stream and self.silence:
            print("You set up silence mode, but stream response still will be printed.")

        full_reply = self._response_handler(response, output_prefix, stream)
        return SendResult(full_reply)

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

    def send(self, message_history: MessageHistory, output_prefix="Assistant：", stream=False,
              response_format:Optional[dict]=None) -> SendResult:
        response_format = response_format if response_format is not None else self.response_format

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
            response_format=response_format
        )

        if stream and self.silence:
            print("You set up silence mode, but stream response still will be printed.")

        full_reply, reasoning_content, logprobs = self._response_handler(response, output_prefix, stream)

        return SendResult(full_reply, reasoning_content, logprobs)

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

        thinking_config = None
        if include_thoughts and self.model_params.model_alias.startswith(self._THINKING_CAPABLE_PREFIX):
            thinking_config = types.ThinkingConfig(include_thoughts=True)

        self._base_config_kwargs = dict(temperature=self.model_params.temperature,
                                        top_p=self.model_params.top_p,
                                        seed=self.model_params.random_seed,
                                        thinking_config=thinking_config)

        # google.genai 的 Chat 对象自己持有整段对话历史，且系统提示只能在创建会话时
        # 设置一次，无法像 OpenAI 那样按次传入。因此这里不能像其它 transient 数据一样
        # 简单地"改成按参数传递"，而是需要让并发的多个 message_history 各自拥有独立的
        # chat 会话，不共享 self.chat，一个 handler 才能安全地被多个线程/Conversation
        # 复用。用 WeakKeyDictionary 是因为 key 是 message_history 本身，其生命周期由
        # 调用方（Conversation）掌控，handler 不必手动清理已结束会话的缓存。
        self._chat_sessions = weakref.WeakKeyDictionary()
        self._chat_sessions_lock = threading.Lock()

    def _get_chat_session(self, message_history: MessageHistory):
        with self._chat_sessions_lock:
            chat = self._chat_sessions.get(message_history)
            if chat is None:
                config = types.GenerateContentConfig(system_instruction=message_history.sys_prompt,
                                                     **self._base_config_kwargs)
                chat = self.client.chats.create(model=self.model_params.model_alias, config=config)
                self._chat_sessions[message_history] = chat
            return chat

    def send(self, message_history:MessageHistory, output_prefix="Assistant：", stream=False,
              response_format:Optional[dict]=None) -> SendResult:
        chat = self._get_chat_session(message_history)

        message = message_history.last_message

        if stream and self.silence:
            print("You set up silence mode, but stream response still will be printed.")

        if stream:
            response = chat.send_message_stream(message)
        else:
            response = chat.send_message(message)

        full_reply, reasoning_content = self._response_handler(response, output_prefix, stream)

        return SendResult(full_reply, reasoning_content, None)

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





