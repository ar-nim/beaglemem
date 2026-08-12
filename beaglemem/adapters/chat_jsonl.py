"""Chat-log corpus: JSONL with role/content rows (Hermes sessions, OpenAI-style
exports). Keeps user + assistant natural text; drops tool rows and tool blocks."""
import glob
import json
import os

from beaglemem.corpus import MIN_SENTENCE_WORDS, split_sentences, tokenize

KEEP_ROLES = {"user", "assistant"}


def _extract_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            b["text"] for b in content
            if isinstance(b, dict) and isinstance(b.get("text"), str)
        )
    return ""


def iter_sentences(path: str):
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "*.jsonl")))
    else:
        files = [path]
    for file_path in files:
        with open(file_path, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if obj.get("role") not in KEEP_ROLES:
                    continue
                text = _extract_text(obj.get("content"))
                if not text:
                    continue
                for raw in split_sentences(text):
                    words = tokenize(raw)
                    if len(words) >= MIN_SENTENCE_WORDS:
                        yield words
