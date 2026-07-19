# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

ChatAPI is a thin Python wrapper around multiple LLM providers (OpenAI-compatible APIs and Google Gemini), used for interactive chat sessions and batch/parallel experiments (e.g. argument-quality annotation, translation). There is no test suite, build system, or dependency manifest (no `requirements.txt`/`pyproject.toml`) — dependencies are installed ad hoc and include `openai`, `google-genai` (`google.genai`), `pandas`, `tqdm`, `pyparsing`.

## Running the code

- Interactive single-session chat: `python main.py` — prompts for a history save path, then runs a REPL loop (`/end` to quit and save).
- Batch/parallel processing: `python main_parallel.py` — reads rows from a `pandas.DataFrame` (currently a stub `pd.DataFrame([])`, meant to be replaced with real input data) and fans requests out via `ThreadPoolExecutor`.
- `Playground.py` / `Playground2.py` are scratch/experiment entry points (git-ignored) — safe to edit freely, not part of the core library.

Provider credentials are read from environment variables per provider: `OPENAI_KEY`, `GOOGLE_KEY`, `DEEPSEEK_KEY` (and ad hoc ones like `NEWAPI_KEY` defined locally in a script, see `Playground2.py`). Set these before running any entry point that talks to a real API.

## Architecture

The core library lives in `chat/`. Object relationships flow as follows:

- **`ConnectionParams`** (`chat/ConnectionParams.py`) — per-provider config: `base_url`, `api_key` (loaded from an env var name passed to `__init__`), and a `model_alias` set by calling one of the provider's `*_model_name()`-style methods (e.g. `OpenAIConnectionParams.gpt_mini()`, `GoogleConnectionParams.gemini_flash()`, `DeepSeekConnectionParams.deepseek_pro()`). Add new providers here as subclasses of `ConnectionParams`.
- **`ModelHyperParams`** (`chat/HyperParams.py`) — carries `model`, `temperature`, `top_p`, `random_seed`. Has `record()`/`from_record()` for (de)serializing into a session file's `metadata.hp`. The class docstring notes: **when adding a new hyperparameter, update `ModelHyperParams` and the corresponding `send()`/`from_hparams()` call sites in `ConnectionHandler`** so the field is actually threaded through to the API call.
- **`ConnectionHandler`** (`chat/ConnectionHandler.py`) — wraps the actual provider SDK call. Base class is a no-op stub; real behavior lives in subclasses:
  - `OpenAIConnectionHandler` — used for any OpenAI-compatible endpoint (OpenAI, DeepSeek, local/NewAPI gateways) by pointing `ConnectionParams.base_url` at the right host. Handles both streaming and non-streaming responses, including `reasoning_content` (CoT) extraction from deltas/messages.
  - `GoogleConnectionHandler` — uses `google.genai`'s chat-session API directly (not the OpenAI-compat endpoint). Lazily (re)creates the underlying `genai` chat session when the system instruction changes, since Gemini's chat API takes system instructions at session-creation time rather than as a message.
  - Both take a `ModelHyperParams` + `ConnectionParams` pair and expose `.send(message_history, output_prefix)`.
- **`MessageHistory`** (`chat/MessageHistory.py`) — owns the session's JSON structure (`version`, `session`, `metadata`, `messages`) and message IDs (`msg_NNN`, `sys_000`). Supports `load()` from a dict/path and round-trips model hyperparameters through `metadata.hp`. This is the canonical on-disk session format (see files under `chat/history/`, which is git-ignored).
- **`Prompt` / `PromptManager`** (`chat/Prompt.py`) — loads prompt templates from `chat/prompts/prompts.json`. Each prompt entry has per-language `system`/`user` text (inline or via `path_system`/`path_user` file references) and a `variables` list; `Prompt.user_message(**kwargs)` does `{{var}}` placeholder substitution. Add new prompts by adding entries to `prompts.json`, not by hardcoding strings in scripts.
- **`Conversation`** (`chat/Conversation.py`) — the orchestrator glueing the above together. `init_session()` sets up a `MessageHistory` from a `Prompt` (system + optional first user message); `send()` appends a user message, delegates to `connection_handler.send()`, then records the assistant reply (plus `reasoning_content` if present) back into history. `save_to()`/`save_to_jsonl()` dump the full session JSON to disk.

Typical wiring (see `main.py`/`Playground.py` for the pattern): pick a `*ConnectionParams` subclass → set its model alias → build a `ModelHyperParams` → construct the matching `*ConnectionHandler` → get a `Prompt` from `PromptManager` → construct a `Conversation` → `init_session(prompt=...)` → `send(text)` in a loop → `save_to(path)`.

`Archive/` contains old, unmaintained one-off scripts kept for reference only.
