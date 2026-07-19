# Roadmap

本文件追踪 ChatAPI 的功能开发进度，作为进度的单一事实来源。每完成一项开发任务，回来更新本文件：把条目从"计划实现"移动到"已实现"并勾选。

## 已实现功能

- [x] 多供应商接入：OpenAI 兼容端点（OpenAI / DeepSeek / NewAPI 网关），Google Gemini 原生 `google.genai` 接口
- [x] `Conversation` 会话编排（`chat/Conversation.py`）：串联 Prompt、历史记录与连接层
- [x] `MessageHistory` 会话 JSON 落盘格式（`chat/MessageHistory.py`）：`version`/`session`/`metadata`/`messages` 结构，支持 `load()`/`save_to()`/`save_to_jsonl()`
- [x] `Prompt` / `PromptManager` 模板机制（`chat/Prompt.py`、`chat/prompts/prompts.json`）：多语言、`{{var}}` 占位符替换
- [x] DeepSeek/OpenAI 兼容端点的 `reasoning_content`（CoT）提取与保存（`OpenAIConnectionHandler._response_handler`）；本地 vLLM 只要配置了 `--reasoning-parser`，也会通过这条同一路径落盘
- [x] Google Gemini 的 CoT 保存（`GoogleConnectionHandler`）：Gemini 把思考内容和正文分别放在 `content.parts` 里，用 `part.thought` 布尔标记区分（不是独立字段，也不是内联标签）。新增 `_split_parts()` 按标记拆分 (正文, 推理)，非流式/流式都适配；只对 `gemini*` 系列模型自动开启 `ThinkingConfig(include_thoughts=True)`，`gemma*` 系列不支持 thinking 故不传该配置以避免报错。已通过离线单测验证（构造假 parts，覆盖非流式/流式、有无 thought part 的场景）
- [x] 交互式 REPL（`main.py`）
- [x] 并行批处理（`main_parallel.py`，`ThreadPoolExecutor` + pandas）
- [x] 本地开源 LLM 后端接入（`chat/ConnectionParams.py`）：`VLLMConnectionParams`（默认 `http://localhost:8000/v1`）、`LlamaCppConnectionParams`（默认 `http://localhost:8080/v1`）、`LMStudioConnectionParams`（默认 `http://localhost:1234/v1`）；`ConnectionParams` 基类在环境变量缺失时 `api_key` 回退占位符 `"EMPTY"`
- [x] vLLM Python 全托管：新建 `chat/VLLMServer.py`，`subprocess` 启动 `vllm serve`，轮询 `/health` 等待就绪（超时抛出并附带子进程输出尾部），支持上下文管理器 + `atexit` 自动清理；`connection_params()` 直接返回配套的 `VLLMConnectionParams`
- [x] 示例脚本 `Playground3.py`：`VLLMServer` → `Conversation` 端到端串联

  > ⚠️ 以上三项代码已写完并通过语法检查，但**未在真实 vLLM/llama.cpp/LM Studio 服务上做过端到端联调**（当前开发环境无 GPU、无 `vllm` 包、沙箱网络受限）。首次使用请在本地跑一遍 `Playground3.py`（替换为可用模型）验证连通性与自动清理行为。

  > ⚠️ Gemini CoT 提取同样**未在真实 Gemini API 上联调过**（沙箱无 `google-genai` 包、网络受限），`ThinkingConfig`/`part.thought` 的具体字段名基于已知的 SDK 用法实现，如果安装的 `google-genai` 版本字段有出入，请对照实际返回结构调整 `GoogleConnectionHandler._split_parts()`。

## 计划实现功能

- [ ] OpenAI o 系列 / GPT-5 系列推理模型的 reasoning summary 保存：这类模型默认不返回推理明文，需切到 Responses API，用 `reasoning: {"summary": "auto"}` 拿摘要，且多轮对话需要传递不透明的 `reasoning` item（加密块），与现有 Chat Completions 路径结构不同，需要单独实现
- [ ] Anthropic Claude extended thinking 的 CoT 保存：`thinking` content block（含 `signature` 字段，多轮传递需原样带回校验），走 Messages API，与 OpenAI 兼容格式不同，需要新增专门的 `ConnectionHandler` 子类。当前实验用不到 Anthropic 模型，暂不实现

## 明确不做（本阶段）

- Ollama 后端
- `ConnectionHandler` 层统一 reasoning 抽象重构
- 本地模型（无原生 reasoning_content 字段）的 CoT 保存：曾实现过 `<think>` 标签解析（非流式正则 + 流式状态机），按需求已回退移除。只保存原生支持 CoT 的模型（DeepSeek、开启 reasoning parser 的本地 vLLM、Gemini）的推理链，纯内联标签的本地模型暂不需要 CoT 落盘