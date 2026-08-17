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

REASONING_PARSER_RULES: list[tuple[re.Pattern, str]] = [
    # 蒸馏模型沿用教师模型的思维链格式，必须排在 qwen/llama 规则之前
    (re.compile(r"deepseek[-_.]?r1", re.I),        "deepseek_r1"),
    (re.compile(r"\bqwq\b", re.I),                 "deepseek_r1"),
    (re.compile(r"qwen3", re.I),                   "qwen3"),
    (re.compile(r"gemma[-_.]?4", re.I),            "gemma4"),
    (re.compile(r"gpt[-_.]?oss", re.I),            "openai_gptoss"),
    (re.compile(r"glm[-_.]?4\.?5", re.I),          "glm45"),
    (re.compile(r"minimax[-_.]?m1", re.I),         "minimax"),
    (re.compile(r"granite", re.I),                 "granite"),
    (re.compile(r"hunyuan", re.I),                 "hunyuan_a13b"),
    (re.compile(r"magistral|ministral|mistral", re.I), "mistral"),
]

class OpenSourceModel:
    def __init__(self, model_id):             # model_id = "mistralai/Ministral-3-14B-Instruct-2512-BF16"
        self.repo = model_id.split("/")[0]
        self.model_alias = model_id.split("/")[1]

    @property
    def size(self) -> str:
        """模型名中的参数量标记，如 '14B'、'8x7B'、'350M'。"""
        m = SIZE_RE.search(self.model_alias)
        if m is None:
            return ""
        moe, num, unit = m.groups()
        return f"{moe or ''}{num}{unit.upper()}"

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
    def reasoning_parser(self) -> str | None:
        """根据模型名返回 vLLM 的 --reasoning-parser 取值；非推理模型返回 None。"""
        for pattern, parser in REASONING_PARSER_RULES:
            if pattern.search(self.model_alias):
                return parser
        return None

    @property
    def vllm_extra_args(self) -> list[str]:
        if self.repo == "mistralai":
            return [
                "--tokenizer-mode", "mistral",
                "--config-format", "mistral",
                "--load-format", "mistral"
            ]
        return []