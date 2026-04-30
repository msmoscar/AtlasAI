# AtlasAI: A high-performance reasoning assistant with memory and web search capabilities
# i think the best model is qwen2.5 or qwen3, but you can use any gguf model you like. best if its a reasoning model.
import json
import math
import os
import pathlib
import re
import requests
import sys
import time
import html
from typing import Any, Dict, List, Optional, Tuple

try:
    import numpy as np
except ImportError:
    raise ImportError("Atlas requires numpy. Install it with: pip install numpy")

try:
    from llama_cpp import Llama
except Exception:
    Llama = None

try:
    from sentence_transformers import SentenceTransformer
    _HAS_SENTENCE_TRANSFORMERS = True
except Exception:
    SentenceTransformer = None  # type: ignore
    _HAS_SENTENCE_TRANSFORMERS = False

try:
    from PySide6.QtCore import Qt, QTimer, Signal, QThread
    from PySide6.QtGui import QAction
    from PySide6.QtWidgets import (
        QApplication,
        QWidget,
        QVBoxLayout,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QPushButton,
        QScrollArea,
        QMenuBar,
        QTextEdit,
        QDialog,
        QComboBox,
        QDialogButtonBox,
        QFileDialog,
        QInputDialog,
    )
    _HAS_QT = True
except Exception:
    Qt = QTimer = Signal = QThread = QApplication = QWidget = QVBoxLayout = QHBoxLayout = QLabel = QLineEdit = QPushButton = QScrollArea = QMenuBar = QAction = QTextEdit = QDialog = QFileDialog = QInputDialog = None
    _HAS_QT = False

# Prefiring the GPU can help reduce latency on the first query, so we do a quick check here to see if we can use it and set the default number of layers accordingly.
# Prefiring n_ctx window to set automatically based on available VRAM or system RAM, with a safety buffer to avoid OOM crashes. This is a best-effort approach and may not be perfect, but it should help optimize the default settings for most users without requiring manual configuration.
def _auto_detect_gpu_layers() -> int:
    env_layers = os.environ.get("ATLASAI_GPU_LAYERS")
    if env_layers:
        try:
            return max(0, int(env_layers))
        except ValueError:
            pass
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True
        )
        free_mb = int(result.stdout.strip())
        layers = max(0, (free_mb - 512) // 150)
        return min(layers, 99)
    except Exception:
        pass
    return 0

def _auto_detect_context_size(safety_buffer_mb: int = 512) -> int:
    try:
        import psutil
        print(f"[Atlas] psutil available: {psutil.virtual_memory().available // (1024*1024)}MB free RAM")
    except Exception as e:
        print(f"[Atlas] psutil branch failed: {e}")
    env_ctx = os.environ.get("ATLASAI_CTX_SIZE")
    if env_ctx:
        try:
            return max(512, int(env_ctx))
        except ValueError:
            pass
    # Try VRAM first (GPU path)
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True
        )
        total_mb, free_mb = (int(x.strip()) for x in result.stdout.strip().split(","))
        # Reserve space for model weights (use total - free as model size estimate)
        model_mb = total_mb - free_mb
        available_mb = free_mb - safety_buffer_mb
        if available_mb > 0:
            #~0.125MB per token per layer for Q4, 28-32 layers typical
            n_layers = 32
            mb_per_token = (n_layers * 2 * 128) / 1024  # key + value, head_dim=128
            max_tokens = int(available_mb / mb_per_token)
            return max(4096, min(max_tokens, 32768))
    except Exception:
        pass
    # Fall back to system RAM (CPU path) - much more conservative
    try:
        import psutil
        free_mb = psutil.virtual_memory().available // (1024 * 1024)
        available_mb = free_mb - safety_buffer_mb
        if available_mb > 0:
            max_tokens = int((available_mb / 0.5) * 1024)
            print(f"[Atlas] calculated max_tokens: {max_tokens}")
            return max(4096, min(max_tokens, 32768))  # cap at 32k for CPU
    except Exception:
        pass
    print("[Atlas] fell through to default context size")
    return 16384

MEMORY_DIR = os.path.join(pathlib.Path.home(), ".AtlasAI")
CHAT_LOG_DIR = os.path.join(MEMORY_DIR, "chats")
MEMORY_FILE = os.path.join(MEMORY_DIR, "memory.jsonl")
CHAT_LOG_FILE = os.path.join(MEMORY_DIR, "chat_history.jsonl")
MODEL_SEARCH_DIR = os.path.expanduser("~/Documents/Ai_Models/")
EMBED_MODEL = "all-MiniLM-L6-v2"
EMBED_DIM = 384
DUCKDUCKGO_API = "https://api.duckduckgo.com/"
DUCKDUCKGO_TIMEOUT = 15
HALF_LIFE_SECONDS = 60 * 60 * 24 * 7
DEFAULT_GPU_LAYERS = _auto_detect_gpu_layers()
ENABLE_AUTO_MEMORY = os.environ.get("ATLASAI_AUTO_MEMORY", "1") == "1"
ENGRAM_TYPE_WEIGHTS = {
    "preference": 1.5,
    "manual": 1.4,
    "web": 1.2,
    "fact": 1.0,
    "event": 1.1,
    "recent": 0.9,
}


def load_markdown_file(filename: str) -> str:
    search_paths = [
        os.path.join(os.getcwd(), filename),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), filename),
    ]
    for path in search_paths:
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    return fh.read().strip()
            except Exception:
                return ""
    return ""


# new system prompt taking from a file instead of hardcoding it here, with a fallback to the old prompt if the file is not found or cannot be read.
SYSTEM_PROMPT = load_markdown_file("system_prompt.md")
# End of system prompt")
# Printing the if it used the default prompt instead of the file-based one, to make it clear to the user what is being used.
if not SYSTEM_PROMPT:
    print("System prompt not found. Please create a file named 'system_prompt.md' in the same directory as Atlas.py with your desired prompt content.")
else:
    print("Loaded system prompt from 'system_prompt.md'.")
PROMPT_TEMPLATE = (
    "{system}\n\n"
    "MEMORY_FILE_SNAPSHOT (authoritative):\n{memory_section}\n\n"
    "RELEVANT_MEMORY_HITS:\n{context}\n\n"
    "{recent_section}"
    "{instructions_section}"
    "{tools_section}"
    "User request:\n{user}\n\n"
    "{web_section}"
    "Assistant:\n"
)

READONLY_COMMANDS = ["!quit", "!exit", "!help", "!memory", "!clear"]
DEFAULT_INSTRUCTIONS_FILENAME = "instructions.md"
DEFAULT_TOOLS_FILENAME = "tools.md"
MAX_PROMPT_MEMORY_ENTRIES = 200
MAX_PROMPT_MEMORY_CHARS = 12000


def find_gguf_models(search_dir: str = MODEL_SEARCH_DIR) -> List[str]:
    if not os.path.isdir(search_dir):
        raise FileNotFoundError(
            f"Model folder not found: {search_dir}. Place your GGUF model there or pass a specific path via --model."
        )

    results: List[str] = []
    for root, _, files in os.walk(search_dir):
        for name in files:
            if name.endswith(".gguf"):
                results.append(os.path.join(root, name))
    results.sort()
    return results


class MemoryStore:
    def __init__(self, path: str, embed_model_name: str = EMBED_MODEL):
        self.path = path
        self.embed_model_name = embed_model_name
        self.entries: List[Dict[str, Any]] = []
        self.embeddings: Optional[np.ndarray] = None
        self.use_fallback = False
        self.model = None
        self.last_error = ""
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self._initialize_embedder()
        self.load()

    def _initialize_embedder(self) -> None:
        if _HAS_SENTENCE_TRANSFORMERS:
            try:
                self.model = SentenceTransformer(self.embed_model_name)
            except Exception:
                self.model = None
                self.use_fallback = True
        else:
            self.use_fallback = True

    def _embed_text(self, texts: List[str]) -> np.ndarray:
        if self.use_fallback or self.model is None:
            return np.vstack([simple_embedding(t) for t in texts]).astype("float32")
        embeddings = self.model.encode(texts, show_progress_bar=False)
        embeddings = np.asarray(embeddings, dtype="float32")
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return embeddings / norms

    def _build_embeddings(self) -> None:
        if not self.entries:
            self.embeddings = None
            return
        texts = [entry["text"] for entry in self.entries]
        self.embeddings = self._embed_text(texts)

    def add(self, text: str, tag: str = "fact", weight: float = 1.0, source: str = "") -> None:
        now = time.time()
        entry = {
            "text": text.strip(),
            "tag": tag,
            "weight": float(weight),
            "source": source,
            "timestamp": now,
        }
        self.entries.append(entry)
    
        # Only embed the new entry and append instead of rebuilding everything
        new_emb = self._embed_text([text.strip()])
        if self.embeddings is None:
            self.embeddings = new_emb
        else:
            self.embeddings = np.vstack([self.embeddings, new_emb])
        
        self.save()

    def search(self, query: str, top_k: int = 4) -> List[str]:
        if not self.entries:
            return []
        query_emb = self._embed_text([query])[0]
        if self.embeddings is None:
            return []
        scores = np.dot(self.embeddings, query_emb)
        now = time.time()
        scored: List[Tuple[float, Dict[str, Any]]] = []
        for i, sim in enumerate(scores.tolist()):
            entry = self.entries[i]
            age = now - entry.get("timestamp", now)
            decay = math.exp(-age / HALF_LIFE_SECONDS)
            type_weight = ENGRAM_TYPE_WEIGHTS.get(entry.get("tag", "fact"), 1.0)
            weight = float(entry.get("weight", 1.0)) * type_weight
            score = float(sim) * weight * decay
            scored.append((score, entry))

        # Dynamic threshold — mean + fraction of std dev
        all_scores = [s for s, _ in scored]
        if all_scores:
            mean = np.mean(all_scores)
            std = np.std(all_scores)
            threshold = max(0.05, mean + 0.3 * std)
        else:
            threshold = 0.05

        scored = [(s, e) for s, e in scored if s > threshold]
        scored.sort(reverse=True, key=lambda item: item[0])
        return [entry["text"] for _, entry in scored[:top_k]]

    def save(self) -> None:
        with open(self.path, "w", encoding="utf-8") as fh:
            for entry in self.entries:
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def load(self) -> None:
        self.entries = []
        if not os.path.exists(self.path):
            return
        with open(self.path, "r", encoding="utf-8") as fh:
            for line in fh:
                try:
                    entry = json.loads(line.strip())
                    if isinstance(entry, dict) and "text" in entry:
                        entry.setdefault("tag", "fact")
                        entry.setdefault("weight", 1.0)
                        entry.setdefault("source", "")
                        entry.setdefault("timestamp", time.time())
                        self.entries.append(entry)
                except json.JSONDecodeError:
                    continue
        self._build_embeddings()


def simple_embedding(text: str) -> np.ndarray:
    text = text.lower().strip()
    if not text:
        return np.zeros(EMBED_DIM, dtype="float32")
    freq: Dict[str, int] = {}
    for i in range(len(text) - 1):
        bg = text[i : i + 2]
        freq[bg] = freq.get(bg, 0) + 1
    items = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:EMBED_DIM]
    vec = np.zeros(EMBED_DIM, dtype="float32")
    for idx, (_, count) in enumerate(items):
        vec[idx] = float(count)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec

def extract_json_text(text: str) -> str:
    """
    Extract the first JSON object or array from the model output.
    Uses a stack to find the matching closing bracket in O(n).
    """
    text = text.strip()
    if not text:
        return ""

    openers = {"{": "}", "[": "]"}
    closers = set(openers.values())

    for start, ch in enumerate(text):
        if ch not in openers:
            continue
        stack = []
        in_string = False
        escape = False
        for end in range(start, len(text)):
            c = text[end]
            if escape:
                escape = False
                continue
            if c == "\\" and in_string:
                escape = True
                continue
            if c == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if c in openers:
                stack.append(openers[c])
            elif c in closers:
                if not stack or stack[-1] != c:
                    break
                stack.pop()
                if not stack:
                    candidate = text[start:end + 1]
                    try:
                        json.loads(candidate)
                        return candidate
                    except json.JSONDecodeError:
                        break

    return ""

def safe_eval_math(expression: str) -> str:
    try:
        expression = expression.strip()
        if not expression:
            return ""
        if re.search(r"[^0-9\.\+\-\*/\(\) \t]", expression):
            return ""
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception:
        return ""


def is_math_query(text: str) -> bool:
    if re.search(r"\d+\s*[\+\-\*/]\s*\d+", text):
        return True
    return False


def format_conversation(history: List[Dict[str, str]]) -> str:
    if not history:
        return "No previous conversation."
    lines = []
    for item in history[-20:]:
        prefix = "User" if item.get("role") == "user" else "Assistant"
        message = item.get("message", "")
        lines.append(f"{prefix}: {message}")
    return "\n".join(lines)


def duckduckgo_search(query: str, max_results: int = 4) -> Dict[str, Any]:
    try:
        params = {
            "q": query,
            "format": "json",
            "no_html": "1",
            "skip_disambig": "1",
            "t": "atlasai",
        }
        response = requests.get(DUCKDUCKGO_API, params=params, timeout=DUCKDUCKGO_TIMEOUT)
        response.raise_for_status()
        data = response.json()
    except Exception as exc:
        return {"query": query, "summary": "", "sources": [], "error": str(exc)}

    summary = data.get("AbstractText", "") or data.get("Heading", "")
    sources: List[Dict[str, str]] = []
    abstract_url = data.get("AbstractURL", "")
    if abstract_url:
        sources.append({"title": data.get("Heading", "DuckDuckGo"), "url": abstract_url})

    topics = data.get("RelatedTopics", [])
    if isinstance(topics, list):
        for topic in topics:
            if len(sources) >= max_results:
                break
            if isinstance(topic, dict):
                if topic.get("Text") and topic.get("FirstURL"):
                    sources.append({"title": topic.get("Text"), "url": topic.get("FirstURL")})
                elif topic.get("Topics"):
                    for sub in topic.get("Topics", []):
                        if len(sources) >= max_results:
                            break
                        if sub.get("Text") and sub.get("FirstURL"):
                            sources.append({"title": sub.get("Text"), "url": sub.get("FirstURL")})

    if not summary and sources:
        summary = sources[0].get("title", "")

    if not summary and not sources:
        summary = "No instant answer available from DuckDuckGo."

    return {"query": query, "summary": summary.strip(), "sources": sources}


class AtlasAI:
    def __init__(self, model_path: Optional[str] = None, memory_path: str = MEMORY_FILE):
        self.model_path = model_path
        self.memory = MemoryStore(memory_path)
        self.history: List[Dict[str, str]] = []
        self.chat_filename: Optional[str] = None
        self.last_prompt = ""
        self.last_raw_response = ""
        self.gpu_layers = int(os.environ.get("ATLASAI_GPU_LAYERS", DEFAULT_GPU_LAYERS))
        self.auto_save_memory = ENABLE_AUTO_MEMORY
        self.llm: Optional[Llama] = None
        if model_path:
            self.llm = self._load_model(model_path)
        self._print_startup_info()

    def _load_model(self, model_path: str) -> Llama:
        if not model_path:
            raise ValueError("Model path is required to load a model.")
        if Llama is None:
            raise ImportError("Atlas requires llama-cpp-python. Install it with: pip install llama-cpp-python")
        os.environ["LLAMA_CUBLAS"] = "1"
        os.environ["GGML_CUBLAS"] = "1"
        os.environ["GGML_CUDA_FORCE_CUBLAS"] = "1"
        os.environ["OMP_NUM_THREADS"] = str(os.cpu_count() or 1)
        os.environ["MKL_NUM_THREADS"] = str(os.cpu_count() or 1)
        os.environ["OPENBLAS_NUM_THREADS"] = str(os.cpu_count() or 1)

        self.gpu_layers = int(os.environ.get("ATLASAI_GPU_LAYERS", DEFAULT_GPU_LAYERS))
        model_kwargs = {
            "model_path": model_path,
            "n_ctx": _auto_detect_context_size(),
            "main_gpu": 0,
            "n_gpu_layers": self.gpu_layers,
            "n_threads": os.cpu_count() or 1,
            "use_mlock": False,
            "use_mmap": True,
            "top_k": 40,
            "top_p": 0.92,
            "temperature": 0.1,
            "repeat_penalty": 1.2,
        }

        return Llama(**model_kwargs)

    def _sanitize_chat_name(self, name: str) -> Optional[str]:
        if not name:
            return None
        cleaned = name.strip()
        if not re.fullmatch(r"[A-Za-z0-9]+(?: [A-Za-z0-9]+){0,2}", cleaned):
            return None
        return cleaned.lower().replace(" ", "_")

    def _derive_chat_name(self) -> str:
        user_messages = [entry["message"] for entry in self.history if entry["role"] == "user"]
        if not user_messages:
            return "last_session"

        candidate = next((msg for msg in user_messages if not msg.strip().startswith("!")), user_messages[0])
        candidate = candidate.strip().lower()
        candidate = re.sub(r"[^a-z0-9 ]+", "", candidate)
        words = [word for word in candidate.split() if word]
        if not words:
            return "last_session"

        return "_".join(words[:4])

    def _ensure_chat_filename(self) -> str:
        if self.chat_filename:
            return self.chat_filename

        base_name = self._derive_chat_name()
        timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        self.chat_filename = f"{base_name}_{timestamp}"
        return self.chat_filename

    def _chat_history_filepath(self, sanitized_name: str) -> str:
        return os.path.join(CHAT_LOG_DIR, f"{sanitized_name}.jsonl")

    def save_chat_history(self, name: Optional[str] = None) -> str:
        os.makedirs(CHAT_LOG_DIR, exist_ok=True)
        if name:
            sanitized = self._sanitize_chat_name(name)
            if not sanitized:
                return "Chat name must be 1-3 words containing only letters and numbers."
            filepath = self._chat_history_filepath(sanitized)
            self.chat_filename = sanitized
        else:
            filename = self._ensure_chat_filename()
            filepath = self._chat_history_filepath(filename)

        with open(filepath, "w", encoding="utf-8") as fh:
            for entry in self.history:
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")

        if name:
            return f"Chat saved as '{name}' in {filepath}."
        return f"Chat saved to {filepath}."

    def load_chat_history(self, name: str) -> str:
        sanitized = self._sanitize_chat_name(name)
        if not sanitized:
            return "Chat name must be 1-3 words containing only letters and numbers."
        filepath = self._chat_history_filepath(sanitized)
        if not os.path.exists(filepath):
            return f"Saved chat '{name}' not found."

        loaded: List[Dict[str, str]] = []
        with open(filepath, "r", encoding="utf-8") as fh:
            for line in fh:
                try:
                    entry = json.loads(line.strip())
                    if isinstance(entry, dict) and "role" in entry and "message" in entry:
                        loaded.append(entry)
                except json.JSONDecodeError:
                    continue

        self.history = loaded
        self.chat_filename = sanitized
        return f"Loaded chat '{name}' with {len(self.history)} messages."

    def load_chat_history_file(self, path: str) -> str:
        if not os.path.exists(path):
            return f"Chat file not found: {path}"

        loaded: List[Dict[str, str]] = []
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                try:
                    entry = json.loads(line.strip())
                    if isinstance(entry, dict) and "role" in entry and "message" in entry:
                        loaded.append(entry)
                except json.JSONDecodeError:
                    continue

        self.history = loaded
        self.chat_filename = os.path.splitext(os.path.basename(path))[0]
        return f"Loaded chat file {os.path.basename(path)} with {len(self.history)} messages."

    def list_chat_history(self) -> str:
        if not os.path.isdir(CHAT_LOG_DIR):
            return "No saved chats found."

        files = [f for f in os.listdir(CHAT_LOG_DIR) if f.endswith(".jsonl")]
        if not files:
            return "No saved chats found."

        lines = ["Saved chats:"]
        for filename in sorted(files):
            name = os.path.splitext(filename)[0].replace("_", " ")
            lines.append(f"- {name}")
        return "\n".join(lines)

    def load_model(self, model_path: str) -> str:
        if not model_path:
            return "Usage: !loadmodel /path/to/model.gguf"
        if os.path.isdir(model_path):
            models = find_gguf_models(model_path)
            if not models:
                return f"No GGUF models found in directory: {model_path}"
            model_path = models[0]

        if not os.path.exists(model_path):
            return f"Model path not found: {model_path}"

        if self.llm is not None:
            self.llm = None
            import gc; gc.collect()

        self.model_path = model_path
        self.llm = self._load_model(model_path)
        return f"Loaded model: {self.model_path}"

    def handle_command(self, user: str) -> Optional[str]:
        lowered = user.lower().strip()
        if lowered == "!help":
            return self._print_help()
        if lowered == "!memory":
            return self.show_memory()
        if lowered == "!clear":
            return self.clear_memory()
        if lowered.startswith("!remember "):
            note = user[len("!remember "):].strip()
            self.memory.add(note, tag="manual")
            return "Saved to memory."
        if lowered.startswith("!savechat "):
            name = user.split(maxsplit=1)[1].strip() if len(user.split(maxsplit=1)) > 1 else ""
            if not name:
                return "Usage: !savechat <name> (1-3 words)"
            return self.save_chat_history(name)
        if lowered == "!savechat":
            return "Usage: !savechat <name> (1-3 words)"
        if lowered == "!listchats":
            return self.list_chat_history()
        if lowered.startswith("!loadchat "):
            name = user.split(maxsplit=1)[1].strip() if len(user.split(maxsplit=1)) > 1 else ""
            if not name:
                return "Usage: !loadchat <name>"
            return self.load_chat_history(name)
        if lowered == "!loadchat":
            return "Usage: !loadchat <name>"
        if lowered == "!chatlog":
            return f"Chat log saved at: {CHAT_LOG_DIR}"
        if lowered.startswith("!loadmodel ") or lowered.startswith("!model "):
            model_path = user.split(maxsplit=1)[1].strip() if len(user.split(maxsplit=1)) > 1 else ""
            return self.load_model(model_path)
        if lowered in ("!loadmodel", "!model"):
            return "Usage: !loadmodel /path/to/model.gguf"
        return None

    def _print_startup_info(self) -> None:
        print("AtlasAI is ready.")
        model_name = self.model_path if self.model_path else "No model loaded"
        print(f"Model: {model_name}")
        print(f"Memory: {self.memory.path}")
        print(f"Memory entries: {len(self.memory.entries)}")
        if self.memory.use_fallback:
            print("Embedding: fallback mode")
        else:
            print(f"Embedding: {self.memory.embed_model_name}")
        if self.gpu_layers > 0:
            print(f"GPU model acceleration enabled with {self.gpu_layers} layer(s) on GPU.")
        else:
            print("GPU model acceleration is disabled or unavailable.")
        print(f"Auto memory saving: {'enabled' if self.auto_save_memory else 'disabled'}")
        if not _HAS_QT:
            print("GUI support unavailable. Install PySide6 or run with --cli for console mode.")
        print("Type '!help' for commands. Start typing your question.")
        print("---")

    def build_prompt(self, user: str, retrieved: List[str], web_summary: str = "", web_sources: str = "") -> str:
        context = "\n".join(retrieved) if retrieved else "No relevant memory found."
        memory_section = self._memory_snapshot_for_prompt()
        recent_context = format_conversation(self.history)
        recent_section = "Recent conversation:\n" + recent_context + "\n\n" if recent_context else ""
        instructions = load_markdown_file(DEFAULT_INSTRUCTIONS_FILENAME)
        tools = load_markdown_file(DEFAULT_TOOLS_FILENAME)
        instructions_section = "Instructions file:\n" + instructions + "\n\n" if instructions else ""
        tools_section = "Tools file:\n" + tools + "\n\n" if tools else ""
        web_section = ""
        if web_summary or web_sources:
            web_section = "Web search summary:\n" + web_summary.strip() + "\n\n"
            if web_sources:
                web_section += "Web sources:\n" + web_sources.strip() + "\n\n"
        return PROMPT_TEMPLATE.format(
            system=SYSTEM_PROMPT,
            memory_section=memory_section,
            context=context,
            user=user,
            web_section=web_section,
            recent_section=recent_section,
            instructions_section=instructions_section,
            tools_section=tools_section,
        )

    def _memory_snapshot_for_prompt(self) -> str:
        if not self.memory.entries:
            return "Memory is empty."

        lines: List[str] = []
        for entry in self.memory.entries[-MAX_PROMPT_MEMORY_ENTRIES:]:
            tag = str(entry.get("tag", "fact"))
            weight = float(entry.get("weight", 1.0))
            text = str(entry.get("text", "")).strip().replace("\n", " ")
            if not text:
                continue
            lines.append(f"- [{tag} w={weight:.2f}] {text}")

        if not lines:
            return "Memory is empty."

        snapshot = "\n".join(lines)
        if len(snapshot) > MAX_PROMPT_MEMORY_CHARS:
            snapshot = snapshot[-MAX_PROMPT_MEMORY_CHARS:]
            first_newline = snapshot.find("\n")
            if first_newline != -1:
                snapshot = snapshot[first_newline + 1 :]
            snapshot = "[Truncated memory snapshot]\n" + snapshot
        return snapshot

    def _should_search(self, user: str) -> bool:
        """Return True if the query is likely to benefit from a web search."""
        lowered = user.lower().strip()
        # Explicit search command
        if lowered.startswith("!search "):
            return True
        # Commands never need searching
        if lowered.startswith("!"):
            return False
        # Recency / news signals
        recency_signals = [
            "latest", "newest", "recent", "right now", "currently", "today",
            "this week", "this month", "this year", "in 2024", "in 2025", "in 2026",
            "news", "update", "current", "now", "just released", "just announced",
            "happening", "live", "breaking",
        ]
        if any(signal in lowered for signal in recency_signals):
            return True
        # Question words about facts that change over time
        factual_patterns = [
            r"\bwho is\b", r"\bwho are\b", r"\bwhat is\b", r"\bwhat are\b",
            r"\bwhere is\b", r"\bwhen (is|was|did|will)\b", r"\bhow (much|many|long|old)\b",
            r"\bprice of\b", r"\bcost of\b", r"\bweather\b", r"\bstock\b",
        ]
        if any(re.search(p, lowered) for p in factual_patterns):
            return True
        return False

    def _prepare_query(self, user: str) -> Tuple[str, str, str]:
        web_summary = ""
        web_sources = ""
        query = user
        if user.lower().startswith("!search "):
            query = user[len("!search "):].strip()
        if self._should_search(user):
            search_data = duckduckgo_search(query)
            web_summary = search_data.get("summary", "")
            sources = search_data.get("sources", [])
            if sources:
                web_sources = "\n".join([f"{src.get('title','')} - {src.get('url','')}" for src in sources])
        return query, web_summary, web_sources

    def _run_query(self, user: str) -> str:
        if self.llm is None:
            raise RuntimeError("No model loaded. Load a GGUF model before running queries.")
        query, web_summary, web_sources = self._prepare_query(user)
        retrieved = self.memory.search(query)
        prompt = self.build_prompt(query, retrieved, web_summary=web_summary, web_sources=web_sources)
        self.last_prompt = prompt
        response = self.llm(
            prompt,
            max_tokens=1024,
            temperature=0.1,
            top_p=0.92,
            repeat_penalty=1.2,
            stop=["\nUser:", "\nAssistant:"],
            stream=False,
        )

        try:
            if isinstance(response, dict):
                choices = response.get("choices")
                if isinstance(choices, list) and choices:
                    text = str(choices[0].get("text", "")).strip()
                else:
                    text = str(response.get("text", "")).strip()
            else:
                text = str(response).strip()
        except Exception as exc:
            text = f"[Error parsing model response: {exc}]"

        self.last_raw_response = text
        return text

    def _split_answer_details(self, text: str) -> Tuple[str, str]:
        text = text.strip()
        markers = ["\nDetails:", "\nThoughts:", "\nReasoning:", "\nThought:", "\nDetail:"]
        for marker in markers:
            if marker in text:
                answer, details = text.split(marker, 1)
                return answer.strip(), details.strip()
        return text, ""

    def respond(self, user: str) -> str:
        if not user:
            return ""
        special = self._handle_special_cases(user)
        if special is not None:
            return special
        raw = self._run_query(user)
        answer, _ = self._split_answer_details(raw)
        answer = self._clean_output(answer)
        if self.auto_save_memory:
            self._auto_save_memory(user, answer)
        return answer

    def respond_with_details(self, user: str) -> Tuple[str, str]:
        if not user:
            return "", ""
        special = self._handle_special_cases(user)
        if special is not None:
            return special, ""
        try:
            raw = self._run_query(user)
            answer, details = self._split_answer_details(raw)
            answer = self._clean_output(answer)
            if self.auto_save_memory:
                try:
                    self._auto_save_memory(user, answer)
                except Exception:
                    pass
            return answer, details
        except Exception as exc:
            return f"Error generating response: {exc}", ""

    def _handle_special_cases(self, user: str) -> Optional[str]:
        lowered = user.lower().strip()
        memory_queries = [
            "what do you remember",
            "show memory",
            "what is in memory",
            "what's in memory",
            "memory entries",
            "saved memory",
            "recall memory",
        ]
        if any(phrase in lowered for phrase in memory_queries):
            return self.show_memory()

        # Intercept explicit save requests so they are guaranteed to persist
        # before the LLM generates any response.
        to_save = self._detect_save_intent(user)
        if to_save:
            tag = "manual"
            if any(w in lowered for w in ["prefer", "like", "love", "hate", "enjoy", "favorite"]):
                tag = "preference"
            elif any(w in lowered for w in ["i am", "i'm", "my name", "i work", "i live"]):
                tag = "fact"
            self.memory.add(to_save, tag=tag, weight=1.5, source="user_request")
            return f"Got it, I'll remember that: {to_save}"

        if is_math_query(user):
            expr_match = re.search(r"([0-9\.\s\+\-\*/\(\)]+)", user)
            if expr_match:
                calc = safe_eval_math(expr_match.group(1))
                if calc:
                    return f"Answer: {calc}"
        return None

    def _clean_output(self, text: str) -> str:
        text = text.strip()
        if not text:
            return "I couldn't generate a response."
        text = re.sub(r"^\s*(assistant:|assistant\n|response:|answer:)\s*", "", text, flags=re.I)
        return text

    def _render_markdown_for_gui(self, text: str) -> str:
        escaped = html.escape(text)
        escaped = re.sub(r"`([^`]+)`", r"<code>\1</code>", escaped)
        escaped = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", escaped)
        escaped = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"<em>\1</em>", escaped)
        return escaped.replace("\n", "<br>")

    def _auto_save_memory(self, user: str, response: str, force_save: bool = False, tag: str = "fact", weight: float = 1.2) -> None:
        if self.llm is None:
            return

        if force_save:
            summarize_prompt = (
                "Summarize the following into a single clean memory entry, max 20 words. "
                "Return only the summary, nothing else.\n\n"
                f"User: {user}\nAssistant: {response}"
            )
            try:
                result = self.llm(summarize_prompt, max_tokens=60, temperature=0.0, stream=False)
                if isinstance(result, dict):
                    summary = result.get("choices", [{"text": ""}])[0].get("text", "").strip()
                else:
                    summary = str(result).strip()
                if summary:
                    self.memory.add(summary, tag=tag, weight=weight, source="user_request")
            except Exception:
                pass
            return

        decision_prompt = (
            "You are a memory curator for AtlasAI. Decide if the following exchange should be saved to long-term memory. "
            "Save only user preferences, stable personal data, recurring goals, important facts, project details, or useful web discoveries. "
            "Do not save casual chit-chat or one-off questions. "
            "Respond with a JSON object exactly like: {{\"save\": true|false, \"summary\": \"...\", \"tag\": \"preference|fact|web|event|manual\"}}."
            "\n\nRecent history:\n{recent}\n\nUser: {user}\nAssistant: {response}\n"
        )
        recent_context = format_conversation(self.history[-20:])
        prompt = decision_prompt.format(recent=recent_context, user=user, response=response)
        try:
            decision_response = self.llm(prompt, max_tokens=180, temperature=0.0, top_p=0.9, stream=False)
            if isinstance(decision_response, dict):
                text = decision_response.get("choices", [{"text": ""}])[0].get("text", "")
            else:
                text = str(decision_response)
            json_text = extract_json_text(text)
            decision = json.loads(json_text) if json_text else {}
        except Exception:
            decision = {}

        save = bool(decision.get("save", False))
        summary = str(decision.get("summary", "")).strip()
        tag = str(decision.get("tag", tag)).strip() or tag
        if save and summary:
            self.memory.add(summary, tag=tag, weight=weight, source="auto")

    def _detect_save_intent(self, user: str) -> Optional[str]:
        """Return the text to save if the user explicitly asks Atlas to remember something."""
        lowered = user.lower().strip()
        explicit_patterns = [
            r"(?:please\s+)?(?:remember|save|note|store|keep track of)\s+(?:that\s+)?(?:my\s+)?(.+)",
            r"(?:i want you to|can you|could you)\s+(?:remember|save|note)\s+(?:that\s+)?(.+)",
            r"(?:make a note|make note)\s+(?:that\s+)?(.+)",
            r"(?:add to memory|save to memory|store in memory)\s*[:\-]?\s*(.+)",
        ]
        for pattern in explicit_patterns:
            m = re.search(pattern, lowered)
            if m:
                captured = m.group(1).strip().rstrip(".!?")
                if len(captured) >= 4:
                    return captured
        return None

    def add_memory_if_relevant(self, user: str, response: str) -> None:
        # Explicit save-intent: extract what to remember and persist it immediately.
        to_save = self._detect_save_intent(user)
        if to_save:
            self.memory.add(to_save, tag="manual", weight=1.5, source="user_request")
            return

        # Passive preference/identity signals.
        lowered = user.lower()
        triggers = ["i prefer", "i like", "my favorite", "i'm", "i am"]
        if any(trigger in lowered for trigger in triggers):
            self.memory.add(user.strip(), tag="preference", weight=1.4, source="passive")

    def show_memory(self) -> str:
        memory_error = getattr(self.memory, "last_error", "")
        if memory_error:
            return f"Memory warning: {memory_error}"
        if not self.memory.entries:
            return "Memory is empty."
        scored = []
        now = time.time()
        for entry in self.memory.entries:
            age = now - entry.get("timestamp", now)
            decay = math.exp(-age / HALF_LIFE_SECONDS)
            type_weight = ENGRAM_TYPE_WEIGHTS.get(entry.get("tag", "fact"), 1.0)
            score = float(entry.get("weight", 1.0)) * type_weight * decay
            scored.append((score, entry))
        scored.sort(reverse=True, key=lambda item: item[0])
        lines = []
        for score, entry in scored[:10]:
            age_hours = int((now - entry.get("timestamp", now)) / 3600)
            lines.append(
                f"- [{entry.get('tag','fact')} w={entry.get('weight',1.0):.2f}] {entry['text']} (age={age_hours}h, score={score:.3f})"
            )
        return "Memory:\n" + "\n".join(lines)

    def clear_memory(self) -> str:
        self.memory.entries = []
        self.memory.embeddings = None
        self.memory.save()
        return "Memory cleared."

    def run(self) -> None:
        while True:
            try:
                user = input("You: ").strip()
            except EOFError:
                print("\nGoodbye.")
                break
            except KeyboardInterrupt:
                print("\nGoodbye.")
                break

            if not user:
                continue
            lower = user.lower()

            if lower in ("!quit", "!exit"):
                print("Goodbye.")
                break

            self.history.append({"role": "user", "message": user})
            command_response = self.handle_command(user)
            if command_response is not None:
                print(command_response)
                self.history.append({"role": "assistant", "message": command_response})
                self.save_chat_history()
                continue

            answer = self.respond(user)
            print(f"Atlas: {answer}\n")
            self.history.append({"role": "assistant", "message": answer})
            self.add_memory_if_relevant(user, answer)
            self.save_chat_history()

    def _print_help(self) -> str:
        return (
            "Commands:\n"
            "  !help         Show this command list\n"
            "  !memory       Show recent memory entries\n"
            "  !clear        Clear all saved memory\n"
            "  !remember X   Save a note to memory\n"
            "  !savechat X   Save the full chat history to disk under name X\n"
            "  !loadchat X   Load a saved chat by name\n"
            "  !listchats    List all saved chats\n"
            "  !chatlog      Show saved chat log location\n"
            "  !loadmodel X  Load a new GGUF model at runtime\n"
            "  !model X      Alias for !loadmodel\n"
            "  !exit         Quit the assistant\n"
        )

if _HAS_QT:
    class ResponseThread(QThread):
        result_ready = Signal(str, str)
        error_occurred = Signal(str)

        def __init__(self, assistant: "AtlasAI", user_text: str):
            super().__init__()
            self.assistant = assistant
            self.user_text = user_text

        def run(self) -> None:
            try:
                answer, details = self.assistant.respond_with_details(self.user_text)
                self.result_ready.emit(answer, details)
            except Exception as exc:
                self.error_occurred.emit(str(exc))

    class ChatBubble(QWidget):
        def __init__(self, assistant: "AtlasAI", role: str, text: str, details: str = ""):
            super().__init__()
            self.assistant = assistant
            layout = QVBoxLayout(self)
            layout.setSpacing(4)

            role_label = QLabel(role)
            role_label.setStyleSheet("color: #94a3b8; font-weight: 700; margin-bottom: 4px;")
            layout.addWidget(role_label)

            for widget in self._render_message(text):
                layout.addWidget(widget)

            if details:
                self.toggle_button = QPushButton("Show details")
                self.details_container = QWidget()
                details_layout = QVBoxLayout(self.details_container)
                details_layout.setContentsMargins(0, 0, 0, 0)
                for widget in self._render_message(details):
                    details_layout.addWidget(widget)
                self.details_container.setVisible(False)
                self.toggle_button.clicked.connect(self._toggle_details)
                layout.addWidget(self.toggle_button)
                layout.addWidget(self.details_container)

            bubble_bg = "#1e293b" if role.lower() == "atlas" else "#111827"
            self.setStyleSheet(
                f"QWidget {{ border-radius: 16px; background: {bubble_bg}; margin: 4px; padding: 10px; }}"
            ) 

        def _render_message(self, text: str) -> list:
            import html as htmllib
            widgets = []
            # Split on code blocks
            parts = re.split(r'(```(?:\w+)?\n?.*?```)', text, flags=re.DOTALL)
            for part in parts:
                code_match = re.match(r'```(\w+)?\n?(.*?)```', part, flags=re.DOTALL)
                if code_match:
                    code = code_match.group(2).rstrip()

                    # Wrapper widget so we can stack the button over the text edit
                    wrapper = QWidget()
                    wrapper.setStyleSheet("QWidget { background: transparent; }")
                    wrapper_layout = QVBoxLayout(wrapper)
                    wrapper_layout.setContentsMargins(0, 0, 0, 0)
                    wrapper_layout.setSpacing(0)

                    # Top bar with copy button aligned right
                    top_bar = QWidget()
                    top_bar.setStyleSheet("QWidget { background: #0d1117; border-radius: 6px 6px 0px 0px; }")
                    top_bar_layout = QHBoxLayout(top_bar)
                    top_bar_layout.setContentsMargins(8, 4, 8, 4)
                    top_bar_layout.addStretch()
                    copy_btn = QPushButton("Copy")
                    copy_btn.setFixedSize(60, 24)
                    copy_btn.setStyleSheet(
                        "QPushButton { background: #334155; color: #94a3b8; border-radius: 4px; font-size: 11px; padding: 0px; }"
                        "QPushButton:hover { background: #475569; color: #f8fafc; }"
                    )
                    copy_btn.clicked.connect(lambda checked, c=code: self._copy_code(c, copy_btn))
                    top_bar_layout.addWidget(copy_btn)
                    wrapper_layout.addWidget(top_bar)

                    # Code text area
                    code_edit = QTextEdit()
                    code_edit.setReadOnly(True)
                    code_edit.setPlainText(code)
                    code_edit.setStyleSheet(
                        "QTextEdit { background-color: #0d1117 !important; color: #c9d1d9 !important; "
                        "font-family: monospace; font-size: 13px; border-radius: 0px 0px 6px 6px; "
                        "border: none; padding: 8px; }"
                    )
                    # Auto-size height to content
                    code_edit.setMinimumHeight(60)
                    code_edit.setMaximumHeight(400)
                    code_edit.document().contentsChanged.connect(
                        lambda: code_edit.setFixedHeight(
                            min(int(code_edit.document().size().height()) + 24, 400)
                        )
                    )
                    wrapper_layout.addWidget(code_edit)
                    widgets.append(wrapper)
                else:
                    if not part.strip():
                        continue
                    rendered = self.assistant._render_markdown_for_gui(part) if hasattr(self.assistant, '_render_markdown_for_gui') else htmllib.escape(part).replace('\n', '<br>')
                    label = QLabel(f"<div style='font-size:14px; color:#e2e8f0; line-height:1.6;'>{rendered}</div>")
                    label.setTextFormat(Qt.RichText)
                    label.setWordWrap(True)
                    label.setTextInteractionFlags(Qt.TextSelectableByMouse | Qt.LinksAccessibleByMouse)
                    widgets.append(label)
            return widgets

        def _copy_code(self, code: str, button: "QPushButton") -> None:
            QApplication.clipboard().setText(code)
            button.setText("Copied!")
            QTimer.singleShot(2000, lambda: button.setText("Copy"))

        def _toggle_details(self) -> None:
            visible = not self.details_container.isVisible()
            self.details_container.setVisible(visible)
            self.toggle_button.setText("Hide details" if visible else "Show details")


    class AtlasGUI(QWidget):
        def __init__(self, assistant: AtlasAI):
            super().__init__()
            self.assistant = assistant
            self.setWindowTitle("Atlas AI")
            self.setGeometry(120, 80, 760, 520)
            self.setMinimumSize(640, 460)
            self.setStyleSheet(
                "QWidget { background: #0f172a; color: #e2e8f0; }"
                "QPushButton { background: #2563eb; color: #f8fafc; border-radius: 8px; padding: 8px 12px; }"
                "QPushButton:hover { background: #3b82f6; }"
                "QLineEdit { background: #1e293b; color: #f8fafc; border: 1px solid #334155; border-radius: 12px; padding: 8px; }"
                "QLabel { color: #e2e8f0; }"
                "QMenuBar { background: #0f172a; color: #cbd5e1; }"
                "QMenuBar::item:selected { background: #334155; }"
            )

            outer_layout = QVBoxLayout(self)
            outer_layout.setContentsMargins(8, 8, 8, 8)
            outer_layout.setSpacing(6)
            header_layout = QHBoxLayout()
            title_label = QLabel("Atlas")
            title_label.setStyleSheet("font-size: 20px; font-weight: 700; color: #f8fafc;")
            subtitle_label = QLabel("meow")
            subtitle_label.setStyleSheet("color: #94a3b8; font-size: 11px;")
            header_layout.addWidget(title_label)
            header_layout.addStretch()
            header_layout.addWidget(subtitle_label)
            outer_layout.addLayout(header_layout)

            menubar = QMenuBar()
            menubar.setMinimumHeight(24)
            file_menu = menubar.addMenu("File")
            self.chat_menu = menubar.addMenu("Chats")
            self.action_save_chat = QAction("Save Chat As...", self)
            self.action_save_chat.triggered.connect(self._on_save_chat_as)
            self.chat_menu.addAction(self.action_save_chat)
            self.action_load_chat = QAction("Load Chat...", self)
            self.action_load_chat.triggered.connect(self._on_load_chat)
            self.chat_menu.addAction(self.action_load_chat)
            self.chat_menu.addSeparator()
            self.action_new_chat = QAction("New Chat", self)
            self.action_new_chat.triggered.connect(self._on_new_chat)
            self.chat_menu.addAction(self.action_new_chat)
            self.chat_menu.addSeparator()
            self.action_save_chat_history = QAction("Save Current Chat", self)
            self.action_save_chat_history.triggered.connect(self._on_save_chat)
            self.chat_menu.addAction(self.action_save_chat_history)
            self.model_menu = menubar.addMenu("Model")
            self._populate_model_menu()
            debug_menu = menubar.addMenu("Debug")
            self.action_show_debug = QAction("Show Debug Panel", self)
            self.action_show_debug.setCheckable(True)
            self.action_show_debug.toggled.connect(self._toggle_debug_panel)
            debug_menu.addAction(self.action_show_debug)
            outer_layout.setMenuBar(menubar)

            self.debug_dialog = QDialog(self)
            self.debug_dialog.setWindowTitle("Atlas Debug")
            self.debug_dialog.resize(700, 500)
            debug_layout = QVBoxLayout(self.debug_dialog)
            self.debug_text = QTextEdit()
            self.debug_text.setReadOnly(True)
            debug_layout.addWidget(self.debug_text)
            self.debug_dialog.setLayout(debug_layout)

            self.scroll_area = QScrollArea()
            self.scroll_area.setWidgetResizable(True)
            self.chat_container = QWidget()
            self.chat_layout = QVBoxLayout(self.chat_container)
            self.chat_layout.setContentsMargins(4, 4, 4, 4)
            self.chat_layout.setSpacing(6)
            self.chat_layout.addStretch()
            self.scroll_area.setWidget(self.chat_container)
            self.scroll_area.setStyleSheet("QScrollArea { border: none; }")

            input_layout = QHBoxLayout()
            input_layout.setContentsMargins(0, 0, 0, 0)
            input_layout.setSpacing(6)
            self.input_line = QLineEdit()
            self.input_line.setPlaceholderText("Ask Atlas...")
            self.input_line.returnPressed.connect(self.on_send)
            self.send_button = QPushButton("Send")
            self.send_button.clicked.connect(self.on_send)
            input_layout.addWidget(self.input_line)
            input_layout.addWidget(self.send_button)

            self.status_label = QLabel("")
            self.status_label.setStyleSheet("color: #94a3b8; font-size: 11px;")

            outer_layout.addWidget(self.scroll_area)
            outer_layout.addLayout(input_layout)
            outer_layout.addWidget(self.status_label)

        def _append_chat(self, role: str, text: str, details: str = "") -> None:
            bubble = ChatBubble(self.assistant, role, text, details)
            self.chat_layout.insertWidget(self.chat_layout.count() - 1, bubble)
            QTimer.singleShot(50, self._scroll_to_bottom)

        def _scroll_to_bottom(self) -> None:
            self.scroll_area.verticalScrollBar().setValue(
                self.scroll_area.verticalScrollBar().maximum()
            )

        def _append_debug(self, text: str) -> None:
            if hasattr(self, 'debug_text') and self.debug_text is not None:
                self.debug_text.append(text)

        def _toggle_debug_panel(self, checked: bool) -> None:
            if checked:
                self.debug_dialog.show()
            else:
                self.debug_dialog.hide()

        def _on_load_model(self) -> None:
            path, _ = QFileDialog.getOpenFileName(
                self,
                "Select GGUF Model",
                MODEL_SEARCH_DIR,
                "GGUF Files (*.gguf);;All Files (*)",
            )
            if not path:
                return
            response = self.assistant.load_model(path)
            self._append_chat("Atlas", response)
            self.assistant.history.append({"role": "assistant", "message": response})
            self.assistant.save_chat_history()

        def _on_unload_model(self) -> None:
            self.assistant.llm = None
            self.assistant.model_path = None
            import gc; gc.collect()
            self.assistant.gpu_layers = 0
            response = "No model loaded. Atlas is now in no-model mode."
            self._append_chat("Atlas", response)
            self.assistant.history.append({"role": "assistant", "message": response})
            self.assistant.save_chat_history()

        def _on_select_model(self, model_path: str) -> None:
            response = self.assistant.load_model(model_path)
            self._append_chat("Atlas", response)
            self.assistant.history.append({"role": "assistant", "message": response})
            self.assistant.save_chat_history()

        def _populate_model_menu(self) -> None:
            self.model_menu.clear()
            unload_action = QAction("Unload model / no model", self)
            unload_action.triggered.connect(self._on_unload_model)
            self.model_menu.addAction(unload_action)
            self.model_menu.addSeparator()
            try:
                models = find_gguf_models()
            except Exception:
                models = []

            if not models:
                action = QAction("No models found", self)
                action.setEnabled(False)
                self.model_menu.addAction(action)
            else:
                for model_path in models:
                    model_name = os.path.basename(model_path)
                    action = QAction(model_name, self)
                    action.setData(model_path)
                    action.triggered.connect(lambda checked, p=model_path: self._on_select_model(p))
                    self.model_menu.addAction(action)
                self.model_menu.addSeparator()
                refresh_action = QAction("Refresh models", self)
                refresh_action.triggered.connect(self._populate_model_menu)
                self.model_menu.addAction(refresh_action)

        def _on_save_chat_as(self) -> None:
            name, ok = QInputDialog.getText(self, "Save Chat As", "Enter chat name (1-3 words):")
            if not ok or not name.strip():
                return
            response = self.assistant.save_chat_history(name.strip())
            self._append_chat("Atlas", response)
            self.assistant.history.append({"role": "assistant", "message": response})

        def _on_save_chat(self) -> None:
            response = self.assistant.save_chat_history()
            self._append_chat("Atlas", response)
            self.assistant.history.append({"role": "assistant", "message": response})

        def _clear_chat_view(self) -> None:
            while self.chat_layout.count():
                item = self.chat_layout.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()
            self.chat_layout.addStretch()

        def _on_new_chat(self) -> None:
            if self.assistant.history:
                try:
                    self.assistant.save_chat_history()
                except Exception:
                    pass

            self.assistant.history = []
            self.assistant.chat_filename = None
            self._clear_chat_view()
            self._append_chat("Atlas", "Started a new chat.")
            self.status_label.setText("New chat ready")

        def _on_load_chat(self) -> None:
            os.makedirs(CHAT_LOG_DIR, exist_ok=True)
            path, _ = QFileDialog.getOpenFileName(
                self,
                "Load Saved Chat",
                CHAT_LOG_DIR,
                "Chat Files (*.jsonl);;All Files (*)",
            )
            if not path:
                return
            response = self.assistant.load_chat_history_file(path)
            self._append_chat("Atlas", response)
            self.assistant.history.append({"role": "assistant", "message": response})

        def on_send(self) -> None:
            user_text = self.input_line.text().strip()
            if not user_text:
                return

            self._append_chat("You", user_text)
            self.current_user_text = user_text
            self.assistant.history.append({"role": "user", "message": user_text})
            self.input_line.clear()
            self.status_label.setText("Thinking...")
            self.send_button.setEnabled(False)
            QApplication.processEvents()

            command_response = self.assistant.handle_command(user_text)
            if command_response is not None:
                self._append_chat("Atlas", command_response)
                self.assistant.history.append({"role": "assistant", "message": command_response})
                self.assistant.save_chat_history()
                self.send_button.setEnabled(True)
                self.status_label.setText("")
                return

            self._append_debug(f"[DEBUG] User input: {user_text}\n")
            self.worker = ResponseThread(self.assistant, user_text)
            self.worker.result_ready.connect(self._on_response_ready)
            self.worker.error_occurred.connect(self._on_response_error)
            self.worker.start()

        def _on_response_ready(self, answer: str, details: str) -> None:
            self.assistant.history.append({"role": "assistant", "message": answer})
            self.assistant.save_chat_history()
            self._append_chat("Atlas", answer, details)
            if self.assistant.last_prompt:
                self._append_debug(f"[DEBUG] Prompt sent:\n{self.assistant.last_prompt}\n")
            if self.assistant.last_raw_response:
                self._append_debug(f"[DEBUG] Raw model response:\n{self.assistant.last_raw_response}\n")
            self.send_button.setEnabled(True)
            self.status_label.setText("")

        def _on_response_error(self, error_message: str) -> None:
            self._append_chat("Atlas", f"Error: {error_message}")
            self.send_button.setEnabled(True)
            self.status_label.setText("")


def select_model(models: List[str], fallback: Optional[str] = None) -> str:
    if not models and not fallback:
        raise FileNotFoundError("No GGUF models found.")
    if fallback:
        return fallback
    if len(models) == 1:
        return models[0]
    print("Available models:")
    for idx, path in enumerate(models, start=1):
        print(f"  {idx}. {os.path.basename(path)}")
    while True:
        choice = input("Choose model number (or press Enter for first): ").strip()
        if not choice:
            return models[0]
        if choice.isdigit() and 1 <= int(choice) <= len(models):
            return models[int(choice) - 1]
        print("Invalid selection.")


def select_model_gui(models: List[str], parent: Optional[QWidget] = None) -> Optional[str]:
    if not _HAS_QT:
        return None

    dialog = QDialog(parent)
    dialog.setWindowTitle("Select Model")
    layout = QVBoxLayout(dialog)
    label = QLabel("Choose a GGUF model or select No model:", dialog)
    layout.addWidget(label)

    combo = QComboBox(dialog)
    combo.addItem("0. No model")
    for idx, path in enumerate(models, start=1):
        combo.addItem(f"{idx}. {os.path.basename(path)}")
    layout.addWidget(combo)

    buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, parent=dialog)
    buttons.accepted.connect(dialog.accept)
    buttons.rejected.connect(dialog.reject)
    layout.addWidget(buttons)

    if dialog.exec() != QDialog.Accepted:
        return None

    index = combo.currentIndex()
    if index == 0:
        return ""
    model_index = index - 1
    if 0 <= model_index < len(models):
        return models[model_index]
    return None


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] in ("-h", "--help"):
        print("Usage: python3 Atlas.py [--model PATH] [--cli]")
        print("       python3 AtlasAI.py [--model PATH] [--cli]")
        return

    model_path = None
    if "--model" in sys.argv:
        idx = sys.argv.index("--model")
        if idx + 1 < len(sys.argv):
            model_path = sys.argv[idx + 1]

    if model_path and not os.path.exists(model_path):
        print(f"Model path not found: {model_path}")
        sys.exit(1)

    if _HAS_QT and "--cli" not in sys.argv:
        app = QApplication(sys.argv)
        app.setStyle("Fusion")
        if model_path is None:
            try:
                models = find_gguf_models()
            except Exception as exc:
                print(f"Error finding models: {exc}")
                sys.exit(1)
            model_path = select_model_gui(models)
            if model_path is None:
                print("Model selection cancelled.")
                sys.exit(0)

        assistant = AtlasAI(model_path=model_path, memory_path=MEMORY_FILE)
        window = AtlasGUI(assistant)
        window.show()
        window.raise_()
        window.activateWindow()
        QTimer.singleShot(0, window.activateWindow)
        sys.exit(app.exec())
    else:
        if model_path is None:
            models = find_gguf_models()
            model_path = select_model(models)
        assistant = AtlasAI(model_path=model_path, memory_path=MEMORY_FILE)
        if not _HAS_QT:
            print("PySide6 is not available, falling back to CLI mode.")
        assistant.run()

if __name__ == "__main__":
    main()