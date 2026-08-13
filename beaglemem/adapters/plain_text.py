"""Plain-text corpus: single file (*.txt, *.md) or directory of them.
Handles BOTH a single file path and a directory path — needed because
on_session_end passes a single archive file path, not a directory."""
import glob
import os

from ..corpus import MIN_SENTENCE_WORDS, split_sentences, tokenize

_TEXT_GLOBS = ("*.txt", "*.md")


def iter_sentences(path: str):
    if os.path.isdir(path):
        files = []
        for pattern in _TEXT_GLOBS:
            files.extend(sorted(glob.glob(os.path.join(path, pattern))))
    elif os.path.isfile(path):
        files = [path]
    else:
        return

    for file_path in files:
        with open(file_path, "r", encoding="utf-8", errors="replace") as fh:
            for raw in split_sentences(fh.read()):
                words = tokenize(raw)
                if len(words) >= MIN_SENTENCE_WORDS:
                    yield words
