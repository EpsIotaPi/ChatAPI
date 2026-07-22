# Roadmap

本文件追踪 ChatAPI 的功能开发进度，按功能类别组织，作为进度的单一事实来源。每完成一项开发任务，回来把对应条目的状态从 `[ ]` 改为 `[x]` 并补充实现说明；新功能先加进对应类别，没有合适类别再新建。未完成条目用 ❗❗ 标注高优先级、❗ 标注中优先级，低优先级不做标注。

## 供应商接入

- [x] OpenAI 兼容端点：OpenAI / DeepSeek / NewAPI 网关（`chat/ConnectionParams.py`、`OpenAIConnectionHandler`）
- [x] Google Gemini 原生接入（`google.genai` 接口，`GoogleConnectionParams`、`GoogleConnectionHandler`）
- [x] 本地开源 LLM 后端：`VLLMConnectionParams`（默认 `http://localhost:8000/v1`）、`LlamaCppConnectionParams`（默认 `http://localhost:8080/v1`）、`LMStudioConnectionParams`（默认 `http://localhost:1234/v1`），均复用 `OpenAIConnectionHandler`；`ConnectionParams` 基类在环境变量缺失时 `api_key` 回退占位符 `"EMPTY"`
- [ ] Ollama 后端 —— 暂不需要

## CoT / 推理链保存

- [x] DeepSeek/OpenAI 兼容端点的 `reasoning_content`（CoT）提取与保存（`OpenAIConnectionHandler._response_handler`）；本地 vLLM 只要配置了 `--reasoning-parser`，也会通过这条同一路径落盘
- [x] Google Gemini 的 CoT 保存（`GoogleConnectionHandler`）：Gemini 把思考内容和正文分别放在 `content.parts` 里，用 `part.thought` 布尔标记区分（不是独立字段，也不是内联标签）。新增 `_split_parts()` 按标记拆分 (正文, 推理)，非流式/流式都适配；只对 `gemini*` 系列模型自动开启 `ThinkingConfig(include_thoughts=True)`，`gemma*` 系列不支持 thinking 故不传该配置以避免报错。已通过离线单测验证，并在真实 Gemini API（`gemini-3-flash-preview`）上联调通过：非流式/流式两种模式下 `reasoning_content` 与 `content` 均正确分离，无残留污染（Gemini 的 thinking 内容默认是英文，即使对话语言是中文，这是模型本身行为，非代码问题）

- [ ] ❗ CoT 统计/分析工具：扫描 `chat/history/` 或批处理产出的 jsonl，统计每次回复的 reasoning 长度、reasoning/正文比例等，用于量化不同供应商的推理链风格差异
- [ ] OpenAI o 系列 / GPT-5 系列推理模型的 reasoning summary 保存：这类模型默认不返回推理明文，需切到 Responses API，用 `reasoning: {"summary": "auto"}` 拿摘要，且多轮对话需要传递不透明的 `reasoning` item（加密块），与现有 Chat Completions 路径结构不同，需要单独实现。暂不需要
- [ ] Anthropic Claude extended thinking 的 CoT 保存：`thinking` content block（含 `signature` 字段，多轮传递需原样带回校验），走 Messages API，与 OpenAI 兼容格式不同，需要新增专门的 `ConnectionHandler` 子类。当前实验用不到 Anthropic 模型，暂不实现
- [ ] 本地模型（无原生 reasoning_content 字段）的 CoT 保存 —— 曾实现过 `<think>` 标签解析（非流式正则 + 流式状态机），按需求已回退移除。只保存原生支持 CoT 的模型（DeepSeek、开启 reasoning parser 的本地 vLLM、Gemini）的推理链，纯内联标签的本地模型暂不需要 CoT 落盘
- [ ] `ConnectionHandler` 层统一 reasoning 抽象重构 —— 暂不需要，各供应商差异较大，维持现状分别实现

## 本地模型托管

- [x] vLLM Python 全托管：新建 `chat/VLLMServer.py`，`subprocess` 启动 `vllm serve`，轮询 `/health` 等待就绪（超时抛出并附带子进程输出尾部），支持上下文管理器 + `atexit` 自动清理；`connection_params()` 直接返回配套的 `VLLMConnectionParams`
- [x] 示例脚本 `Playground3_vllm.py`：`VLLMServer` → `Conversation` 端到端串联。已在远程服务器真实环境联调通过（`Qwen/Qwen2.5-3B-Instruct`，12GB 显卡）：默认 `--gpu-memory-utilization 0.9` 会在 CUDA Graph 捕获阶段 OOM（`torch.AcceleratorError: CUDA error: out of memory`，权重本身能装下，是捕获阶段的额外显存需求超出预留池），通过 `extra_args=["--gpu-memory-utilization", "0.85", "--max-model-len", "4096"]` 收窄显存预留后解决；非推理模型跑通了普通对话保存
- [x] 运行过程反馈：`VLLMServer._wait_until_ready()` 目前轮询 `/health` 期间完全没有输出，模型加载/CUDA Graph 捕获可能耗时几分钟，用户不知道是在正常等待还是卡住了。需要在轮询循环里加进度提示（比如每次 `poll_interval` 打印一次"等待 vLLM 就绪...已等待 Xs"，或把子进程 stdout 实时转发到前台而不是只在失败时打印尾部）

- [ ] ❗ vLLM + CoT 联调测试：目前只验证了普通模型（`Qwen2.5-3B-Instruct`，无 `reasoning_content`）的连通性，还没有用支持推理的模型（如 `Qwen/QwQ-32B` 等）配合 `--reasoning-parser` 实测过 `reasoning_content` 能否正确落盘。需要挑一个显存能放下的推理模型，加上对应的 `--reasoning-parser` 参数跑一遍 `Playground3_vllm.py`，确认走的 `OpenAIConnectionHandler` 路径解析正常
- [ ] ❗ vLLM 健康检查/复用已有实例：`VLLMServer.start()` 目前每次都会新起一个进程；加一个"先探测目标端口是否已有 vLLM 在跑，有则直接复用、没有才启动"的逻辑，省去连续跑多个脚本时反复加载模型的等待时间

## 会话与存储

- [x] `Conversation` 会话编排（`chat/Conversation.py`）：串联 Prompt、历史记录与连接层
- [x] `MessageHistory` 会话 JSON 落盘格式（`chat/MessageHistory.py`）：`version`/`session`/`metadata`/`messages` 结构，支持 `load()`/`save_to()`/`save_to_jsonl()`
- [ ] ❗ 会话加载与续聊（resume）入口：`MessageHistory.load()`/`Conversation.from_history()` 已经实现，但缺一个现成的脚本/CLI 入口——"读取已保存的 session 文件 → 继续对话"，目前只能手写代码调用

## Prompt 管理

- [x] `Prompt` / `PromptManager` 模板机制（`chat/Prompt.py`、`chat/prompts/prompts.json`）：多语言、`{{var}}` 占位符替换
- [x] `PROMPTS_DIR` 环境变量：`PromptManager` 默认加载包内 `chat/prompts/prompts.json`，若设置了 `PROMPTS_DIR` 则改为加载 `$PROMPTS_DIR/prompts.json`，让打包后下游项目可以整套替换成自己的 prompt 库而不用改代码

## 运行入口 / 批处理

- [x] 交互式 REPL（`main.py`）
- [x] 并行批处理（`main_parallel.py`，`ThreadPoolExecutor` + pandas）
- [ ] ❗ 多供应商/多模型 A/B 批量对比脚本：同一个 prompt 在 DeepSeek/Gemini/本地模型等多个供应商上分别跑一遍，对比 CoT 与答案质量，直接复用现有 `ConnectionHandler` 抽象和 `main_parallel.py` 的并行模式

## 工程基础设施

- [x] 依赖管理与打包（`pyproject.toml`）：`chat` 作为可安装包（`chatapi`），锁定 `openai`/`google-genai`/`pandas`/`tqdm`/`pyparsing` 的大版本区间；`chat/prompts/*.json`、`*.txt` 通过 `package-data` 随包分发；版本号单一来源于 `pyproject.toml`，`chat/__init__.py` 通过 `importlib.metadata` 读取暴露为 `__version__`。支持 `pip install git+https://git.epsiotapi.com/EpsIotaPi/ChatAPI.git`（含 `@<tag>` 锁版本）从私有 Git 服务器安装到其它项目/远程服务器。副带修复：`PromptManager` 默认路径及 `prompts.json` 里的 `path_system`/`path_user` 原本是 CWD 相对路径（且有 `chat/prompt/` 目录名 typo），装成包后从别的项目调用会 `FileNotFoundError`；改为基于包内路径解析，已在临时 venv 中验证跨目录导入正常
