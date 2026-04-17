import json
import math
import os
import pathlib
import re
import requests
import sys
import time
import datetime
from typing import Any, Dict, List, Optional

try:
    import numpy as np
except ImportError:
    raise ImportError("AtlasAI.py requires numpy. Install it with: pip install numpy")

try:
    from llama_cpp import Llama
except ImportError:
    raise ImportError("AtlasAI.py requires llama-cpp-python. Install it with: pip install llama-cpp-python")

try:
    from sentence_transformers import SentenceTransformer
    _HAS_SENTENCE_TRANSFORMERS = True
except Exception:
    SentenceTransformer = None  # type: ignore
    _HAS_SENTENCE_TRANSFORMERS = False

try:
    from PySide6.QtCore import Qt, QTimer, Signal, QThread
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
        QAction,
        QTextEdit,
        QDialog,
    )
    _HAS_QT = True
except Exception:
    Qt = QTimer = Signal = QThread = QApplication = QWidget = QVBoxLayout = QHBoxLayout = QLabel = QLineEdit = QPushButton = QScrollArea = QMenuBar = QAction = QTextEdit = QDialog = None
    _HAS_QT = False

MEMORY_DIR = os.path.join(pathlib.Path.home(), ".AtlasAI")
MEMORY_FILE = os.path.join(MEMORY_DIR, "memory.jsonl")
MODEL_SEARCH_DIR = os.path.expanduser("~/Documents/Ai_Models/")
EMBED_MODEL = "all-MiniLM-L6-v2"
EMBED_DIM = 384
DUCKDUCKGO_API = "https://api.duckduckgo.com/"
DUCKDUCKGO_TIMEOUT = 15
HALF_LIFE_SECONDS = 60 * 60 * 24 * 7
ENGRAM_TYPE_WEIGHTS = {
    "preference": 1.5,
    "manual": 1.4,
    "web": 1.2,
    "fact": 1.0,
    "event": 1.1,
    "recent": 0.9,
}

SYSTEM_PROMPT = (
    "You are Atlas, a high-performance reasoning assistant. "
    "Your answers should feel polished, confident, and trustworthy like a leading AI service. "
    "Always prioritize accuracy, avoid hallucination, and keep your response concise. "
    "When solving problems, think through the steps carefully and then respond with a short answer followed by an optional brief explanation. "
    "If you are uncertain, say That you don't know rather than inventing details."
)

PROMPT_TEMPLATE = (
    "{system}\n\n"
    "Context:\n{context}\n\n"
    "{recent_section}"
    "User request:\n{user}\n\n"
    "{web_section}"
    "Instructions:\n"
    "- Provide the final result under 'Answer:' in 1-3 sentences.\n"
    "- If useful, include a brief 'Details:' section after the answer.\n"
    "- Do not expose your internal reasoning unless the user explicitly asks for it.\n"
    "- Prefer clarity and avoid unnecessary repetition.\n"
)

READONLY_COMMANDS = ["!quit", "!exit", "!help", "!memory", "!clear"]


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
        self._build_embeddings()
        self.save()

    def search(self, query: str, top_k: int = 4) -> List[str]:
        if not self.entries:
            return []
        query_emb = self._embed_text([query])[0]
        if self.embeddings is None:
            return []
        scores = np.dot(self.embeddings, query_emb)
        now = time.time()
        scored: List[tuple[float, Dict[str, Any]]] = []
        for i, sim in enumerate(scores.tolist()):
            entry = self.entries[i]
            age = now - entry.get("timestamp", now)
            decay = math.exp(-age / HALF_LIFE_SECONDS)
            type_weight = ENGRAM_TYPE_WEIGHTS.get(entry.get("tag", "fact"), 1.0)
            weight = float(entry.get("weight", 1.0)) * type_weight
            score = float(sim) * weight * decay
            if score > 0.0:
                scored.append((score, entry))

        scored.sort(reverse=True, key=lambda item: item[0])
        results: List[str] = []
        for score, entry in scored[:top_k]:
            results.append(entry["text"])
        return results

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
    """
    text = text.strip()
    if not text:
        return ""

    for start in range(len(text)):
        if text[start] not in "{[":
            continue
        for end in range(start + 1, len(text) + 1):
            candidate = text[start:end]
            try:
                json.loads(candidate)
                return candidate
            except json.JSONDecodeError:
                continue

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
        prefix = "User" if item["role"] == "user" else "Assistant"
        lines.append(f"{prefix}: {item['message']}")
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
    def __init__(self, model_path: str, memory_path: str = MEMORY_FILE):
        self.model_path = model_path
        self.memory = MemoryStore(memory_path)
        self.history: List[Dict[str, str]] = []
        self.last_prompt = ""
        self.last_raw_response = ""
        self.llm = self._load_model(model_path)
        self._print_startup_info()

    def _load_model(self, model_path: str) -> Llama:
        os.environ["LLAMA_CUBLAS"] = "1"
        os.environ["GGML_CUDA_FORCE_CUBLAS"] = "1"
        os.environ["OMP_NUM_THREADS"] = str(os.cpu_count() or 1)
        os.environ["MKL_NUM_THREADS"] = str(os.cpu_count() or 1)
        os.environ["OPENBLAS_NUM_THREADS"] = str(os.cpu_count() or 1)

        return Llama(
            model_path=model_path,
            n_ctx=8192,
            n_gpu_layers=-1,
            n_threads=os.cpu_count() or 1,
            use_mlock=True,
            use_mmap=False,
            top_k=40,
            top_p=0.92,
            temperature=0.1,
            repeat_penalty=1.05,
        )

    def _print_startup_info(self) -> None:
        print("AtlasAI is ready.")
        print(f"Model: {self.model_path}")
        print(f"Memory: {self.memory.path}")
        print(f"Memory entries: {len(self.memory.entries)}")
        if self.memory.use_fallback:
            print("Embedding: fallback mode")
        else:
            print(f"Embedding: {self.memory.embed_model_name}")
        print("Type '!help' for commands. Start typing your question.")
        print("---")

    def build_prompt(self, user: str, retrieved: List[str], web_summary: str = "", web_sources: str = "") -> str:
        context = "\n".join(retrieved) if retrieved else "No relevant memory found."
        recent_context = format_conversation(self.history)
        recent_section = "Recent conversation:\n" + recent_context + "\n\n" if recent_context else ""
        web_section = ""
        if web_summary or web_sources:
            web_section = "Web search summary:\n" + web_summary.strip() + "\n\n"
            if web_sources:
                web_section += "Web sources:\n" + web_sources.strip() + "\n\n"
        return PROMPT_TEMPLATE.format(
            system=SYSTEM_PROMPT,
            context=context,
            user=user,
            web_section=web_section,
            recent_section=recent_section,
        )

    def _prepare_query(self, user: str) -> tuple[str, str, str]:
        web_summary = ""
        web_sources = ""
        query = user
        if user.lower().startswith("!search "):
            query = user[len("!search "):].strip()
            search_data = duckduckgo_search(query)
            web_summary = search_data.get("summary", "")
            sources = search_data.get("sources", [])
            if sources:
                web_sources = "\n".join([f"{src.get('title','')} - {src.get('url','')}" for src in sources])
            if web_summary:
                self.memory.add(
                    f"Web search: {query} | {web_summary}",
                    tag="web",
                    weight=1.2,
                    source="duckduckgo",
                )
        return query, web_summary, web_sources

    def _run_query(self, user: str) -> str:
        query, web_summary, web_sources = self._prepare_query(user)
        retrieved = self.memory.search(query)
        prompt = self.build_prompt(query, retrieved, web_summary=web_summary, web_sources=web_sources)
        self.last_prompt = prompt
        response = self.llm(prompt, max_tokens=512, temperature=0.1, top_p=0.92, stream=False)

        if isinstance(response, dict):
            choices = response.get("choices")
            if isinstance(choices, list) and choices:
                text = choices[0].get("text", "").strip()
            else:
                text = response.get("text", "").strip()
        else:
            text = str(response).strip()

        self.last_raw_response = text
        return text

    def _split_answer_details(self, text: str) -> tuple[str, str]:
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
        self._auto_save_memory(user, answer)
        return answer

    def respond_with_details(self, user: str) -> tuple[str, str]:
        if not user:
            return "", ""
        special = self._handle_special_cases(user)
        if special is not None:
            return special, ""
        try:
            raw = self._run_query(user)
            answer, details = self._split_answer_details(raw)
            answer = self._clean_output(answer)
            try:
                self._auto_save_memory(user, answer)
            except Exception:
                pass
            return answer, details
        except Exception as exc:
            return f"Error generating response: {exc}", ""

    def _handle_special_cases(self, user: str) -> Optional[str]:
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

    def _auto_save_memory(self, user: str, response: str) -> None:
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
        tag = str(decision.get("tag", "fact")).strip() or "fact"
        if save and summary:
            self.memory.add(summary, tag=tag, weight=1.2, source="auto")

    def add_memory_if_relevant(self, user: str, response: str) -> None:
        lowered = user.lower()
        triggers = ["remember that", "i prefer", "i like", "my favorite", "i'm", "i am"]
        if any(trigger in lowered for trigger in triggers):
            text = f"User said: {user} | Assistant responded: {response}"
            self.memory.add(text, tag="preference", weight=1.5)

    def show_memory(self) -> str:
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
            if lower == "!help":
                self._print_help()
                continue
            if lower == "!memory":
                print(self.show_memory())
                continue
            if lower == "!clear":
                print(self.clear_memory())
                continue
            if lower.startswith("!remember "):
                note = user[len("!remember "):].strip()
                self.memory.add(note, tag="manual")
                print("Saved to memory.")
                continue

            answer = self.respond(user)
            print(f"Atlas: {answer}\n")
            self.history.append({"role": "user", "message": user})
            self.history.append({"role": "assistant", "message": answer})
            self.add_memory_if_relevant(user, answer)

    def _print_help(self) -> None:
        print(
            "Commands:\n"
            "  !help       Show this command list\n"
            "  !memory     Show recent memory entries\n"
            "  !clear      Clear all saved memory\n"
            "  !remember X Save a note to memory\n"
            "  !exit       Quit the assistant\n"
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
        def __init__(self, role: str, text: str, details: str = ""):
            super().__init__()
            layout = QVBoxLayout(self)
            role_label = QLabel(f"<b>{role}</b>")
            role_label.setTextFormat(Qt.RichText)
            message_label = QLabel(text.replace("\n", "<br>"))
            message_label.setWordWrap(True)
            layout.addWidget(role_label)
            layout.addWidget(message_label)
            if details:
                self.toggle_button = QPushButton("Show details")
                self.details_label = QLabel(details.replace("\n", "<br>"))
                self.details_label.setWordWrap(True)
                self.details_label.setVisible(False)
                self.toggle_button.clicked.connect(self._toggle_details)
                layout.addWidget(self.toggle_button)
                layout.addWidget(self.details_label)
            self.setStyleSheet(
                "QWidget { border: 1px solid #888; border-radius: 8px; margin: 6px; padding: 8px; }"
            )

        def _toggle_details(self) -> None:
            visible = not self.details_label.isVisible()
            self.details_label.setVisible(visible)
            self.toggle_button.setText("Hide details" if visible else "Show details")


    class AtlasGUI(QWidget):
        def __init__(self, assistant: AtlasAI):
            super().__init__()
            self.assistant = assistant
            self.setWindowTitle("Atlas AI")
            self.setGeometry(120, 80, 940, 700)

            outer_layout = QVBoxLayout(self)
            menubar = QMenuBar()
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
            self.chat_layout.setContentsMargins(8, 8, 8, 8)
            self.chat_layout.setSpacing(8)
            self.chat_layout.addStretch()
            self.scroll_area.setWidget(self.chat_container)

            input_layout = QHBoxLayout()
            self.input_line = QLineEdit()
            self.input_line.setPlaceholderText("Type your message and press Enter...")
            self.input_line.returnPressed.connect(self.on_send)
            self.send_button = QPushButton("Send")
            self.send_button.clicked.connect(self.on_send)
            input_layout.addWidget(self.input_line)
            input_layout.addWidget(self.send_button)

            self.status_label = QLabel("")
            self.status_label.setStyleSheet("color: #555;")

            outer_layout.addWidget(self.scroll_area)
            outer_layout.addLayout(input_layout)
            outer_layout.addWidget(self.status_label)

        def _append_chat(self, role: str, text: str, details: str = "") -> None:
            bubble = ChatBubble(role, text, details)
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

        def on_send(self) -> None:
            user_text = self.input_line.text().strip()
            if not user_text:
                return

            self._append_chat("You", user_text)
            self.current_user_text = user_text
            self.input_line.clear()
            self.status_label.setText("Thinking...")
            self.send_button.setEnabled(False)
            QApplication.processEvents()

            lower = user_text.lower()
            if lower in ("!help", "!memory", "!clear"):
                if lower == "!help":
                    self._append_chat("System", self.assistant._print_help.__doc__ or "Available commands: !help, !memory, !clear, !remember <note>")
                elif lower == "!memory":
                    self._append_chat("System", self.assistant.show_memory())
                elif lower == "!clear":
                    self._append_chat("System", self.assistant.clear_memory())
                self.send_button.setEnabled(True)
                self.status_label.setText("")
                return

            self._append_debug(f"[DEBUG] User input: {user_text}\n")
            self.worker = ResponseThread(self.assistant, user_text)
            self.worker.result_ready.connect(self._on_response_ready)
            self.worker.error_occurred.connect(self._on_response_error)
            self.worker.start()

        def _on_response_ready(self, answer: str, details: str) -> None:
            user_text = getattr(self, 'current_user_text', '')
            if user_text:
                self.assistant.history.append({"role": "user", "message": user_text})
            self.assistant.history.append({"role": "assistant", "message": answer})
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


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] in ("-h", "--help"):
        print("Usage: python3 AtlasAI.py [--model PATH] [--cli]")
        return

    model_path = None
    if "--model" in sys.argv:
        idx = sys.argv.index("--model")
        if idx + 1 < len(sys.argv):
            model_path = sys.argv[idx + 1]

    if model_path and not os.path.exists(model_path):
        print(f"Model path not found: {model_path}")
        sys.exit(1)

    if model_path is None:
        models = find_gguf_models()
        model_path = select_model(models)

    assistant = AtlasAI(model_path=model_path, memory_path=MEMORY_FILE)
    if _HAS_QT and "--cli" not in sys.argv:
        app = QApplication(sys.argv)
        window = AtlasGUI(assistant)
        window.show()
        sys.exit(app.exec())
    else:
        assistant.run()


if __name__ == "__main__":
    main()
