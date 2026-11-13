"""Turns text into numbers. Two backends sit behind one interface."""

import hashlib
import re

import numpy as np

from config import config

_WORD = re.compile(r"[a-z0-9]+")
_STOPWORDS = set(config.OFFLINE_EMBEDDING_STOPWORDS)


class Embedder(object):
    """The interface: encode many, encode one, describe, and a name."""

    name = "base"

    def encode(self, texts):
        raise NotImplementedError

    def encode_one(self, text):
        return self.encode([text])[0]

    def describe(self):
        return {"backend": self.name}


def _stem(word):
    # A deliberately small stemmer: plural and common verb endings only, so
    # "returns", "returned" and "return" land on the same hash position.
    for suffix in ("ing", "ed", "es", "s"):
        if len(word) > len(suffix) + 3 and word.endswith(suffix) and not word.endswith("ss"):
            return word[: -len(suffix)]
    return word


def _position(term, dimensions):
    digest = hashlib.md5(term.encode("utf-8")).hexdigest()
    return int(digest, 16) % dimensions


class OfflineEmbedder(Embedder):
    """A deterministic hashed bag of words and word pairs, numpy only.

    Rarer terms weigh more, approximated by term length because the backend holds
    no corpus state. It matches on shared vocabulary, not on meaning.
    """

    name = "offline_hashing"

    def __init__(self, dimensions=None):
        self.dimensions = config.OFFLINE_EMBEDDING_DIMENSIONS if dimensions is None else dimensions

    def _terms(self, text):
        words = [_stem(word) for word in _WORD.findall((text or "").lower())
                 if word not in _STOPWORDS]
        return words

    def _vector(self, text):
        vector = np.zeros(self.dimensions, dtype=np.float64)
        words = self._terms(text)
        for word in words:
            vector[_position(word, self.dimensions)] += np.log1p(len(word))
        for first, second in zip(words, words[1:]):
            pair = first + " " + second
            vector[_position(pair, self.dimensions)] += (
                config.OFFLINE_BIGRAM_WEIGHT * np.log1p(len(pair)))
        norm = np.linalg.norm(vector)
        return vector / norm if norm > 0 else vector

    def encode(self, texts):
        return np.vstack([self._vector(text) for text in texts]) if texts else \
            np.zeros((0, self.dimensions))

    def describe(self):
        return {"backend": self.name, "dimensions": self.dimensions}


class MiniLMEmbedder(Embedder):
    """all-MiniLM-L6-v2 through sentence-transformers, when it is installed."""

    name = "minilm"

    def __init__(self):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(config.EMBEDDING_MODEL_NAME,
                                         device=config.EMBEDDING_DEVICE)

    def encode(self, texts):
        return np.asarray(self.model.encode(list(texts), normalize_embeddings=True))

    def describe(self):
        return {"backend": self.name, "model": config.EMBEDDING_MODEL_NAME,
                "device": config.EMBEDDING_DEVICE}


def minilm_is_available():
    """Whether the real embedder can be used, and why not when it cannot."""
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        return False, "sentence-transformers is not installed (pip install -r requirements-models.txt)"
    return True, "sentence-transformers is installed"


def load_embedder(backend=None):
    """'offline' insists on the stand-in, 'minilm' on the real model, 'auto' prefers the real one."""
    choice = (config.EMBEDDING_BACKEND if backend is None else backend).strip().lower()
    if choice in ("offline", OfflineEmbedder.name):
        return OfflineEmbedder()
    if choice in ("minilm",):
        available, reason = minilm_is_available()
        if not available:
            raise RuntimeError("The minilm embedder was requested and is unavailable: " + reason)
        return MiniLMEmbedder()
    if choice == "auto":
        available, reason = minilm_is_available()
        if available:
            try:
                embedder = MiniLMEmbedder()
                print("Embedding backend: minilm")
                return embedder
            except Exception as error:  # a download failure must not stop the platform
                reason = "the model could not be loaded: " + str(error)
        print("Embedding backend: offline_hashing (" + reason + ")")
        return OfflineEmbedder()
    raise ValueError("Unknown embedding backend " + repr(choice)
                     + "; use 'minilm', 'offline' or 'auto'.")


def embedder_for_backend_name(name):
    """The embedder that matches a backend name recorded on an index."""
    if name == MiniLMEmbedder.name:
        return load_embedder("minilm")
    return load_embedder("offline")
