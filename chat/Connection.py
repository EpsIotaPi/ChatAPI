import os

class ConnectionParams:
    base_url: str
    api_key: str

    def __init__(self, key_env: str):
        self.api_key = str(os.getenv(key_env))



class OpenAIConnection(ConnectionParams):
    base_url: str = "https://api.openai.com/v1"
    api_key: str

    def __init__(self, key_env: str = "OPENAI_KEY"):
        super().__init__(key_env)
        self.api_key = str(os.getenv(key_env))
        self.gpt = "gpt-5.5"
        self.gpt_mini = "gpt-5.4-mini"

class GoogleConnection(ConnectionParams):
    base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    api_key: str

    def __init__(self, key_env: str = "GOOGLE_KEY"):
        super().__init__(key_env)
        self.api_key = str(os.getenv(key_env))
        self.gemma = "gemma-4-31b"
        self.gemini_pro = "gemini-3.1-pro-preview"
        self.gemini_flash = "gemini-3-flash-preview"
        self.gemini_flash_lite = "gemini-3.1-flash-lite"

class DeepSeekConnection(ConnectionParams):
    base_url: str = "https://api.deepseek.com"
    api_key: str

    def __init__(self, key_env: str = "DEEPSEEK_KEY"):
        super().__init__(key_env)
        self.api_key = str(os.getenv(key_env))
        self.deepseek_flash = "deepseek-v4-flash"
        self.deepseek_pro = "deepseek-v4-pro"
