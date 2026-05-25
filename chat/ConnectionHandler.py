from openai import OpenAI
from chat.HyperParams import ModelHyperParams
from chat.Connection import ConnectionParams


class ConnectionHandler:
    def __init__(self, model_params:ModelHyperParams, connection_params:ConnectionParams,
                 silence=False, stream=False):
        self.model_params = model_params
        self.connection_params = connection_params

        self.silence = silence
        self.stream = stream
        self.response_format = {"type": "text"}

    @classmethod
    def from_hparams(cls, model:str, connection: ConnectionParams,
                     temperature = None, top_p = None, random_seed = None, **kwargs):

        hp = ModelHyperParams(model=model, temperature=temperature, top_p=top_p, random_seed=random_seed)
        return cls(hp, connection, **kwargs)

    def send(self, message, output_prefix="Assistant："):
        """
        把 message 与 hp 整理成 send_content，并向LLM请求 response，用 response_handler 处理成 full_reply
        :param message:
        :param output_prefix:
        :return:
        """
        send_content = {
            "messages": message,
            "model": self.model_params.model,
            "temperature": self.model_params.model,
            "top_p": self.model_params.top_p,
            "seed": self.model_params.random_seed,
            "stream": self.stream,
            "response_format": self.response_format
        }

        response = "this is the response from LLM"

        full_reply = self._response_handler(response)
        return full_reply

    def set_silence_mode(self):
        self.silence = True

    def _response_handler(self, response, output_prefix="Assistant："):
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

    def _print(self, obj, end: str | None = "\n", ):
        if not self.silence:
            print(obj, end=end)


class OpenAIConnectionHandler(ConnectionHandler):
    def __init__(self, model_params:ModelHyperParams, connection_params:ConnectionParams,
                 silence=False, stream=False):
        super().__init__(model_params, connection_params, silence, stream)

        self.client = OpenAI(api_key=connection_params.api_key,
                             base_url=connection_params.base_url)

    def send(self, message, output_prefix="Assistant："):
        response = self.client.chat.completions.create(
            messages=message,
            model=self.model_params.model,
            temperature=self.model_params.temperature,
            top_p=self.model_params.top_p,
            seed=self.model_params.random_seed,
            stream=self.stream,
            response_format=self.response_format
        )
        full_reply = self._response_handler(response, output_prefix)

        return full_reply

    def _response_handler(self, response, output_prefix="Assistant："):
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

        return full_reply





