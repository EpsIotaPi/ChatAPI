import os

class ConnectionParams:
    base_url: str
    api_key: str

    def __init__(self, key_env: str):
        self.api_key = str(os.getenv(key_env))
        self.model_alias = "default_model"

class OpenAIConnectionParams(ConnectionParams):
    base_url: str = "https://api.openai.com/v1"
    api_key: str

    def __init__(self, key_env: str = "OPENAI_KEY"):
        super().__init__(key_env)
        self.api_key = str(os.getenv(key_env))
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
        self.api_key = str(os.getenv(key_env))
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
        self.api_key = str(os.getenv(key_env))
        self.deepseek_flash()

    def deepseek_flash(self):
        self.model_alias = "deepseek-v4-flash"

    def deepseek_pro(self):
        self.model_alias = "deepseek-v4-pro"
