import os

class ConnectionParams:
    base_url: str
    api_key: str

    def __init__(self, key_env: str):
        # 本地推理服务（vLLM/llama.cpp/LM Studio 等）通常不校验 api_key，
        # 环境变量缺失时回退到占位符，避免把字符串 "None" 当成真实 key 发出去。
        self.api_key = os.getenv(key_env, "EMPTY")
        self.model_alias = "default_model"

class OpenAIConnectionParams(ConnectionParams):
    base_url: str = "https://api.openai.com/v1"
    api_key: str

    def __init__(self, key_env: str = "OPENAI_KEY"):
        super().__init__(key_env)
        self.gpt_mini(version="5.4")

    def gpt(self, version="5.5"):
        self.model_alias = f"gpt-{version}"

    def gpt_mini(self, version="5.4"):
        self.model_alias = f"gpt-{version}-mini"

class GoogleConnectionParams(ConnectionParams):
    base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    api_key: str

    def __init__(self, key_env: str = "GOOGLE_KEY"):
        super().__init__(key_env)
        self.gemma()

    def gemma(self):
        self.model_alias = "gemma-4-31b"

    def gemini_pro(self):
        self.model_alias = "gemini-3.1-pro-preview"

    def gemini_flash(self):
        self.model_alias = "gemini-3-flash-preview"

    def gemini_flash_lite(self):
        self.model_alias = "gemini-3.1-flash-lite"

class DeepSeekConnectionParams(ConnectionParams):
    base_url: str = "https://api.deepseek.com"
    api_key: str

    def __init__(self, key_env: str = "DEEPSEEK_KEY"):
        super().__init__(key_env)
        self.deepseek_flash()

    def deepseek_flash(self):
        self.model_alias = "deepseek-v4-flash"

    def deepseek_pro(self):
        self.model_alias = "deepseek-v4-pro"


class VLLMConnectionParams(ConnectionParams):
    """
    接入本地 vLLM 的 OpenAI 兼容端点（默认由 `vllm serve` 起在 8000 端口）。
    本地服务通常不校验 api_key，缺省环境变量时使用占位符 "EMPTY"。
    """
    base_url: str = "http://localhost:8000/v1"
    api_key: str

    def __init__(self, key_env: str = "VLLM_KEY", host: str = "localhost", port: int = 8000):
        super().__init__(key_env)
        self.base_url = f"http://{host}:{port}/v1"
        self.model_alias = "default_model"

    def set_model(self, model_name: str):
        """vLLM 直接用启动时加载的模型路径/名作为请求里的 model 字段。"""
        self.model_alias = model_name


class LlamaCppConnectionParams(ConnectionParams):
    """接入本地 llama.cpp server 的 OpenAI 兼容端点（默认 8080 端口）。"""
    base_url: str = "http://localhost:8080/v1"
    api_key: str

    def __init__(self, key_env: str = "LLAMACPP_KEY", host: str = "localhost", port: int = 8080):
        super().__init__(key_env)
        self.base_url = f"http://{host}:{port}/v1"
        self.model_alias = "default_model"

    def set_model(self, model_name: str):
        self.model_alias = model_name


class LMStudioConnectionParams(ConnectionParams):
    """接入本地 LM Studio 的 OpenAI 兼容端点（默认 1234 端口）。"""
    base_url: str = "http://localhost:1234/v1"
    api_key: str

    def __init__(self, key_env: str = "LMSTUDIO_KEY", host: str = "localhost", port: int = 1234):
        super().__init__(key_env)
        self.base_url = f"http://{host}:{port}/v1"
        self.model_alias = "default_model"

    def set_model(self, model_name: str):
        self.model_alias = model_name
