import atexit
import collections
import subprocess
import threading
import time
import urllib.error
import urllib.request

from chatapi.ConnectionParams import VLLMConnectionParams


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
                 extra_args: list | None = None, startup_timeout: float = 300.0,
                 poll_interval: float = 2.0, verbose: bool = True):
        self.model = model
        self.host = host
        self.port = port
        self.extra_args = extra_args or []
        self.startup_timeout = startup_timeout
        self.poll_interval = poll_interval
        self.verbose = verbose

        self._process = None
        self._atexit_registered = False
        self._output_buffer = collections.deque(maxlen=200)
        self._reader_thread = None

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}/v1"

    @property
    def health_url(self) -> str:
        return f"http://{self.host}:{self.port}/health"

    def start(self):
        if self._process is not None and self._process.poll() is None:
            return self  # 已经在运行，幂等

        if self.verbose:
            self._print_separator()
            print(f"正在加载模型：{self.model} ...")

        cmd = ["vllm", "serve", self.model, "--host", self.host, "--port", str(self.port)]
        cmd.extend(self.extra_args)

        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        self._reader_thread = threading.Thread(target=self._drain_output, daemon=True)
        self._reader_thread.start()

        if not self._atexit_registered:
            atexit.register(self.stop)
            self._atexit_registered = True

        self._wait_until_ready()
        return self

    def _drain_output(self):
        # 只收集到缓冲区供失败时兜底展示，不实时打印，避免刷屏。
        for line in self._process.stdout:
            self._output_buffer.append(line.rstrip("\n"))

    def _wait_until_ready(self):
        display_tick = 1.0
        start = time.monotonic()
        deadline = start + self.startup_timeout
        next_check = start

        while time.monotonic() < deadline:
            if self._process.poll() is not None:
                if self.verbose:
                    print()
                raise VLLMServerError(
                    f"vllm serve 进程提前退出（returncode={self._process.returncode}）。\n"
                    f"{self._tail_output()}"
                )

            now = time.monotonic()
            if now >= next_check:
                try:
                    with urllib.request.urlopen(self.health_url, timeout=self.poll_interval) as resp:
                        if resp.status == 200:
                            if self.verbose:
                                print(f"\r模型加载成功（耗时 {time.monotonic() - start:.1f} 秒）")
                                self._print_separator()
                            return
                except (urllib.error.URLError, ConnectionError, TimeoutError):
                    pass
                next_check = now + self.poll_interval

            if self.verbose:
                elapsed = time.monotonic() - start
                print(f"\r已等待 {elapsed:.0f}/{self.startup_timeout:.0f} 秒", end="", flush=True)
            time.sleep(display_tick)

        if self.verbose:
            print()
        self.stop()
        raise VLLMServerError(
            f"等待 vLLM 就绪超时（{self.startup_timeout}s），已终止子进程。\n{self._tail_output()}"
        )

    def _tail_output(self, n_lines: int | None = None) -> str:
        lines = list(self._output_buffer) if n_lines is None else list(self._output_buffer)[-n_lines:]
        return f"vllm serve 完整输出（最多保留最近 {self._output_buffer.maxlen} 行）：\n" + "\n".join(lines)

    @staticmethod
    def _print_separator():
        print("=" * 60)

    def stop(self):
        if self._process is None or self._process.poll() is not None:
            return

        if self.verbose:
            self._print_separator()
            print("正在关闭...")

        self._process.terminate()
        try:
            self._process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait(timeout=10)

        if self._reader_thread is not None:
            self._reader_thread.join(timeout=5)

        if self.verbose:
            print("模型已关闭")
            self._print_separator()

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
