# Roadmap

本文件追踪 ChatAPI 的功能开发进度，作为进度的单一事实来源。每完成一项开发任务，回来更新本文件：把条目从"计划实现"移动到"已实现"并勾选。

## 已实现功能

- [x] 多供应商接入：OpenAI 兼容端点（OpenAI / DeepSeek / NewAPI 网关），Google Gemini 原生 `google.genai` 接口
- [x] `Conversation` 会话编排（`chat/Conversation.py`）：串联 Prompt、历史记录与连接层
- [x] `MessageHistory` 会话 JSON 落盘格式（`chat/MessageHistory.py`）：`version`/`session`/`metadata`/`messages` 结构，支持 `load()`/`save_to()`/`save_to_jsonl()`
- [x] `Prompt` / `PromptManager` 模板机制（`chat/Prompt.py`、`chat/prompts/prompts.json`）：多语言、`{{var}}` 占位符替换
- [x] DeepSeek/OpenAI 兼容端点的 `reasoning_content`（CoT）提取与保存（`OpenAIConnectionHandler._response_handler`）
- [x] 交互式 REPL（`main.py`）
- [x] 并行批处理（`main_parallel.py`，`ThreadPoolExecutor` + pandas）
- [x] 本地开源 LLM 后端接入（`chat/ConnectionParams.py`）：`VLLMConnectionParams`（默认 `http://localhost:8000/v1`）、`LlamaCppConnectionParams`（默认 `http://localhost:8080/v1`）、`LMStudioConnectionParams`（默认 `http://localhost:1234/v1`）；`ConnectionParams` 基类在环境变量缺失时 `api_key` 回退占位符 `"EMPTY"`
- [x] 本地模型 CoT 保存增强：`OpenAIConnectionHandler._response_handler` 支持解析内联 `<think>...</think>` 标签（非流式用 `_split_think_tags` 正则；流式用跨 chunk 状态机 `_partial_tag_suffix_len`），仅在无独立 `reasoning_content` 字段时启用，不影响 DeepSeek 路径。已通过离线单测验证（构造假响应，覆盖非流式/流式、独立字段/内联标签/无 reasoning 三种场景）
- [x] vLLM Python 全托管：新建 `chat/VLLMServer.py`，`subprocess` 启动 `vllm serve`，轮询 `/health` 等待就绪（超时抛出并附带子进程输出尾部），支持上下文管理器 + `atexit` 自动清理；`connection_params()` 直接返回配套的 `VLLMConnectionParams`
- [x] 示例脚本 `Playground3.py`：`VLLMServer` → `Conversation` 端到端串联

  > ⚠️ 以上三项代码已写完并通过语法检查/离线单测，但**未在真实 vLLM/llama.cpp/LM Studio 服务上做过端到端联调**（当前开发环境无 GPU、无 `vllm` 包、沙箱网络受限）。首次使用请在本地跑一遍 `Playground3.py`（替换为可用模型）验证连通性与自动清理行为。

## 计划实现功能

（暂无，如后续验证发现问题或有新需求，在此追加）

## 明确不做（本阶段）

- Ollama 后端
- Gemini thinking 的 CoT 提取
- `ConnectionHandler` 层统一 reasoning 抽象重构