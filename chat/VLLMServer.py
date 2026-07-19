import atexit
import subprocess
import time
import urllib.error
import urllib.request

from chat.ConnectionParams import VLLMConnectionParams


class VLLMServerError(Exception):
    pass


class VLLMServer:
    """
    在 Python 进程内全托管一个本地 vLLM 服务：负责启动 `vllm serve`、
    轮询 /health 等待就绪、以及在退出时（正常/异常/上下文管理器）自动关闭子进程。

    用法：
        with VLLMServer("Qwen/Qwen2.5-7B-Instruct") as srv:
            connection_params = srv.connection_params()
            handler = OpenAIConnectionHandler(hp, connection_params, stream=True)
            ...
    """

    def __init__(self, model: str, host: str = "127.0.0.1", port: int = 8000,
                 extra_args: list = None, startup_timeout: float = 300.0,
                 poll_interval: float = 2.0):
        self.model = model
        self.host = host
        self.port = port
        self.extra_args = extra_args or []
        self.startup_timeout = startup_timeout
        self.poll_interval = poll_interval

        self._process = None
        self._atexit_registered = False

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}/v1"

    @property
    def health_url(self) -> str:
        return f"http://{self.host}:{self.port}/health"

    def start(self):
        if self._process is not None and self._process.poll() is None:
            return self  # 已经在运行，幂等

        cmd = ["vllm", "serve", self.model, "--host", self.host, "--port", str(self.port)]
        cmd.extend(self.extra_args)

        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        if not self._atexit_registered:
            atexit.register(self.stop)
            self._atexit_registered = True

        self._wait_until_ready()
        return self

    def _wait_until_ready(self):
        deadline = time.monotonic() + self.startup_timeout
        while time.monotonic() < deadline:
            if self._process.poll() is not None:
                raise VLLMServerError(
                    f"vllm serve 进程提前退出（returncode={self._process.returncode}）。\n"
                    f"{self._tail_output()}"
                )
            try:
                with urllib.request.urlopen(self.health_url, timeout=self.poll_interval) as resp:
                    if resp.status == 200:
                        return
            except (urllib.error.URLError, ConnectionError, TimeoutError):
                pass
            time.sleep(self.poll_interval)

        self.stop()
        raise VLLMServerError(
            f"等待 vLLM 就绪超时（{self.startup_timeout}s），已终止子进程。\n{self._tail_output()}"
        )

    def _tail_output(self, n_lines: int = 20) -> str:
        if self._process is None or self._process.stdout is None:
            return ""
        try:
            output = self._process.stdout.read()
        except Exception:
            return ""
        lines = output.splitlines()[-n_lines:]
        return "vllm serve 输出尾部：\n" + "\n".join(lines)

    def stop(self):
        if self._process is None or self._process.poll() is not None:
            return
        self._process.terminate()
        try:
            self._process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait(timeout=10)

    def connection_params(self, key_env: str = "VLLM_KEY") -> VLLMConnectionParams:
        """返回指向本实例地址的 VLLMConnectionParams，并把 model_alias 设为启动时的模型名。"""
        params = VLLMConnectionParams(key_env=key_env, host=self.host, port=self.port)
        params.set_model(self.model)
        return params

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False
