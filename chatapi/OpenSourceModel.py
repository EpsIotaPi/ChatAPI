import re

SIZE_RE = re.compile(
    r"(?<![\w.])"  # 左边不能是字母/数字/点
    r"(\d+x)?"  # 可选的 MoE 前缀，如 8x
    r"(\d+(?:\.\d+)?)"  # 主体数字，支持 1.5
    r"([BbMm])"  # 单位
    r"(?![\w])",  # 右边不能紧跟字母数字
)

QUANT_PATTERNS: list[tuple[str, re.Pattern]] = [
    # llama.cpp / GGUF 风格：Q4_K_M、Q5_0、IQ3_XXS
    ("gguf", re.compile(r"(?<![\w])(I?Q\d(?:_[A-Z0-9]+)+)(?![\w])", re.I)),
    # 权重/激活位宽：W8A8、W4A16
    ("wa", re.compile(r"(?<![\w])(W\d+A\d+)(?![\w])", re.I)),
    # 浮点：BF16、FP16、FP8、FP32
    ("float", re.compile(r"(?<![\w])((?:BF|FP)\d{1,2})(?![\w])", re.I)),
    # 整数：INT8、INT4、Int4
    ("int", re.compile(r"(?<![\w])(INT\d{1,2})(?![\w])", re.I)),
    # 裸位宽：4bit、8-bit
    ("bit", re.compile(r"(?<![\w])(\d{1,2})[-_ ]?bit(?![\w])", re.I)),
    # 方法名：AWQ、GPTQ、GGUF、EXL2、NF4、HQQ
    ("method", re.compile(r"(?<![\w])(AWQ|GPTQ|GGUF|EXL2|NF4|HQQ|SmoothQuant)(?![\w])", re.I)),
]

# 名称片段 -> vLLM `--reasoning-parser` 取值。
# NOTE: 这些取值需与目标 vLLM 版本的 `vllm serve --help` 核对，不同版本命名可能变动。
REASONING_PARSER_RULES: list[tuple[re.Pattern, str]] = [
    # 蒸馏模型沿用教师模型的思维链格式，必须排在 qwen/llama 规则之前
    (re.compile(r"deepseek[-_.]?r1", re.I),        "deepseek_r1"),
    (re.compile(r"\bqwq\b", re.I),                 "deepseek_r1"),
    (re.compile(r"qwen3", re.I),                   "qwen3"),
    (re.compile(r"gpt[-_.]?oss", re.I),            "openai_gptoss"),
    (re.compile(r"glm[-_.]?4\.?5", re.I),          "glm45"),
    (re.compile(r"minimax[-_.]?m1", re.I),         "minimax"),
    (re.compile(r"granite", re.I),                 "granite"),
    (re.compile(r"hunyuan", re.I),                 "hunyuan_a13b"),
    (re.compile(r"magistral|ministral|mistral", re.I), "mistral"),
]

# 名称片段 -> vLLM `--tool-call-parser` 取值。
# NOTE: 同样需与目标 vLLM 版本核对；多数 Llama 系微调走 hermes 模板。
TOOL_PARSER_RULES: list[tuple[re.Pattern, str]] = [
    (re.compile(r"qwen3", re.I),                   "hermes"),
    (re.compile(r"qwen2\.?5|qwen2", re.I),         "hermes"),
    (re.compile(r"hermes", re.I),                  "hermes"),
    (re.compile(r"deepseek[-_.]?v3|deepseek[-_.]?r1", re.I), "deepseek_v3"),
    (re.compile(r"granite", re.I),                 "granite"),
    (re.compile(r"llama[-_.]?3", re.I),            "llama3_json"),
    (re.compile(r"magistral|ministral|mistral", re.I), "mistral"),
]

# quant_size 的 method 标记 -> vLLM `--quantization` 取值。
_METHOD_QUANT_MAP = {
    "AWQ": "awq",
    "GPTQ": "gptq",
    "GGUF": "gguf",
    "EXL2": "exl2",
    "NF4": "bitsandbytes",
    "HQQ": "hqq",
    "SMOOTHQUANT": "compressed-tensors",
}


class OpenSourceModel:
    """
    从 HuggingFace 风格的 model_id 解析出模型的元信息，并汇总成一份可直接喂给
    `vllm serve` 的启动参数（见 to_vllm_args）。

    名称能推断的字段（参数量 / 量化 / reasoning-parser / tool-call-parser / dtype）
    自动解析；部署相关或需要覆盖的字段通过关键字入参显式给出，**显式值优先于名称派生**。

    预期联动用法（本模块不直接改动 VLLMServer）：
        model = OpenSourceModel("Qwen/Qwen3-235B-A22B-Instruct", tensor_parallel_size=8)
        with VLLMServer(model.model_id, extra_args=model.to_vllm_args()) as srv:
            ...
    """

    tensor_parallel_size: int | None = None
    gpu_memory_utilization: float | None = None
    max_model_len: int | None = None
    trust_remote_code: bool = False
    served_model_name: str | None = None
    chat_template: str | None = None
    kv_cache_dtype: str | None = None

    def __init__(self, model_id, quantization: str | None = None, dtype: str | None = None):
        # model_id 可能是 "org/Model"、裸模型名，或本地路径 "/data/models/org/Model"。
        # 只取最后一段作为 model_alias，其余作为 repo，避免 split("/")[1] 在无 "/" 时越界。
        self.model_id = model_id
        parts = model_id.rstrip("/").rsplit("/", 1)
        if len(parts) == 2:
            self.repo, self.model_alias = parts
        else:
            self.repo, self.model_alias = "", parts[0]

        self._quantization_override = quantization
        self._dtype_override = dtype

    @property
    def size(self) -> str:
        """模型名中的参数量标记，如 '14B'、'8x7B'、'350M'。"""
        m = SIZE_RE.search(self.model_alias)
        if m is None:
            return ""
        moe, num, unit = m.groups()
        return f"{(moe or '').lower()}{num}{unit.upper()}"

    @property
    def quant_size(self) -> dict[str, str]:
        """模型名中的量化相关标记，返回 {类别: 标记} 字典。"""
        found: dict[str, str] = {}
        for kind, pattern in QUANT_PATTERNS:
            m = pattern.search(self.model_alias)
            if m is not None:
                found[kind] = m.group(1).upper()
        return found

    @property
    def quantization(self) -> str | None:
        """
        vLLM `--quantization` 取值：显式入参优先，否则从 quant_size 派生。
        浮点标记（BF16/FP16）不算量化方法，交给 --dtype 处理；FP8 例外，属量化。
        """
        if self._quantization_override is not None:
            return self._quantization_override
        quant = self.quant_size
        if "method" in quant:
            return _METHOD_QUANT_MAP.get(quant["method"])
        if "gguf" in quant:
            return "gguf"
        if quant.get("float", "").startswith("FP8"):
            return "fp8"
        if "wa" in quant or "int" in quant:
            # W8A8 / INT8 之类通常由 compressed-tensors 格式承载
            return "compressed-tensors"
        return None

    @property
    def dtype(self) -> str | None:
        """vLLM `--dtype` 取值：显式入参优先，否则从 BF16/FP16 标记派生；无则 None（auto）。"""
        if self._dtype_override is not None:
            return self._dtype_override
        fp = self.quant_size.get("float", "")
        if fp == "BF16":
            return "bfloat16"
        if fp == "FP16":
            return "float16"
        if fp == "FP32":
            return "float32"
        return None

    @property
    def reasoning_parser(self) -> str | None:
        """根据模型名返回 vLLM 的 --reasoning-parser 取值；非推理模型返回 None。"""
        for pattern, parser in REASONING_PARSER_RULES:
            if pattern.search(self.model_alias):
                return parser
        return None

    @property
    def tool_call_parser(self) -> str | None:
        """根据模型名返回 vLLM 的 --tool-call-parser 取值；无匹配返回 None。"""
        for pattern, parser in TOOL_PARSER_RULES:
            if pattern.search(self.model_alias):
                return parser
        return None

    @property
    def extra_args(self) -> list[str]:
        """
        厂商 / 权重格式特有的启动参数（不含通用调优项，那些在 to_vllm_args 中汇总）。
        mistral 三件套仅适用于 Mistral 官方原生权重；社区量化仓库（如 TheBloke/*）
        的 repo 名不同，不会命中，这是刻意的——量化版不能用 mistral load-format。
        """
        if self.repo == "mistralai":
            return [
                "--tokenizer-mode", "mistral",
                "--config-format", "mistral",
                "--load-format", "mistral",
            ]
        return []

    @property
    def vllm_args(self) -> list[str]:
        """
        汇总全部 `vllm serve` 启动参数（不含模型名本身），顺序稳定，便于：
            VLLMServer(model.model_id, extra_args=model.to_vllm_args())
        """
        args: list[str] = []

        # —— 模型固有：推理 / 工具解析器 ——
        reasoning_parser = self.reasoning_parser
        if reasoning_parser:
            args += ["--reasoning-parser", reasoning_parser]
        tool_call_parser = self.tool_call_parser
        if tool_call_parser:
            args += ["--enable-auto-tool-choice", "--tool-call-parser", tool_call_parser]

        # —— 量化 / 精度 ——
        quantization = self.quantization
        if quantization:
            args += ["--quantization", quantization]
        dtype = self.dtype
        if dtype:
            args += ["--dtype", dtype]
        if self.kv_cache_dtype:
            args += ["--kv-cache-dtype", self.kv_cache_dtype]

        # —— 加载兼容 ——
        if self.trust_remote_code:
            args += ["--trust-remote-code"]
        if self.served_model_name:
            args += ["--served-model-name", self.served_model_name]
        if self.chat_template:
            args += ["--chat-template", self.chat_template]

        # —— 运行资源 ——
        if self.tensor_parallel_size is not None:
            args += ["--tensor-parallel-size", str(self.tensor_parallel_size)]
        if self.gpu_memory_utilization is not None:
            args += ["--gpu-memory-utilization", str(self.gpu_memory_utilization)]
        if self.max_model_len is not None:
            args += ["--max-model-len", str(self.max_model_len)]

        args += self.extra_args
        return args
