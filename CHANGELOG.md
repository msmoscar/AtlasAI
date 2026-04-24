# Changelog

All notable changes to Atlas will be documented here.

## [0.1.0-alpha] — 2026-04-24

### Added
- Local GGUF model support via `llama-cpp-python` with automatic GPU layer detection
- Persistent memory system with semantic search and time-decay scoring
- Memory commands: `!remember`, `!memory`, `!clear`
- DuckDuckGo web search integration with automatic query detection
- CLI mode
- GUI mode (PySide6) with dark theme, chat bubbles, and markdown rendering
- Code blocks with copy button in GUI
- Model switcher — load/unload GGUF models at runtime
- Named session save/load
- Debug panel for raw prompt inspection
- Graceful fallbacks for optional dependencies (`sentence-transformers`, `PySide6`, `torch`)
