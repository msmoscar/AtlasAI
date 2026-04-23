# AtlasAI
A local AI assistant with a PySide6 GUI and CLI fallback for GGUF models.

## Overview

AtlasAI is a single-file Python assistant that can run as a desktop GUI or console chatbot. It supports:
- Local GGUF LLM models via `llama-cpp-python`
- Persistent memory storage for context and recall
- DuckDuckGo web search integration with optional `!search` commands
- Automatic preference and fact memory saving
- A polished Gemini-inspired chat UI when PySide6 is installed
- CLI fallback when GUI mode is unavailable

## Requirements

- Python 3.10+
- `PySide6` for GUI mode
- `llama-cpp-python`
- `numpy`
- `requests`
- `sentence-transformers`
- `psutil`

Install dependencies:

```bash
pip install -r requirements.txt
```

## Running AtlasAI

By default, AtlasAI attempts to launch the GUI if PySide6 is installed. If GUI support is unavailable, it falls back to the CLI.

```bash
python3 Atlas.py
```

Legacy entrypoint (supported via wrapper):

```bash
python3 AtlasAI.py
```

Force CLI mode:

```bash
python3 Atlas.py --cli
```

Use a specific GGUF model:

```bash
python3 Atlas.py --model /path/to/model.gguf
```

## Commands

- `!help` — show available commands
- `!memory` — display current saved memory entries
- `!clear` — clear saved memory
- `!remember <note>` — save a manual note to memory
- `!savechat <name>` — save the current chat to a named file (1-3 words)
- `!loadchat <name>` — load a previously saved chat by name
- `!listchats` — list all saved chats
- `!chatlog` — show the saved chat directory location
- `!loadmodel <path>` — hot swap to a new GGUF model without restarting
- `!model <path>` — alias for `!loadmodel`
- `!search <query>` — perform a DuckDuckGo search and add the result to memory

Atlas now automatically includes web search results for normal queries, so it can use fresh information when needed.

- Place `instructions.md` in the repo root to give Atlas per-prompt instructions.
- Place `tools.md` in the repo root to document available tools and APIs.

## Notes

- Memory is stored at `~/.AtlasAI/memory.jsonl` by default.
- If the default location is not writable, Atlas falls back to `./.AtlasAI/memory.jsonl` in the current working directory.
- You can force a specific memory location with `ATLASAI_MEMORY_DIR`, for example:

```bash
ATLASAI_MEMORY_DIR=/path/to/writable/folder python3 Atlas.py
```
- Default search path for models is `~/Documents/Ai_Models/`
- The GUI includes a File menu for loading models, saving named chats, and loading saved chats
- Named chat files are stored in `~/.AtlasAI/chats/` and use 1-3 word names
- The GUI window includes a debug panel for prompt and model output inspection

## Troubleshooting

- If the GUI does not start, confirm `PySide6` is installed:

```bash
pip install PySide6
```

- If you see a model path error, pass `--model /path/to/model.gguf` or place models in `~/Documents/Ai_Models/`
