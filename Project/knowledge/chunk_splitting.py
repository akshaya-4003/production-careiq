"""Cuts a document into the units retrieval searches, at the document's own headings."""

import re

from config import config


def looks_like_heading(line):
    """A short capitalised line with no terminal punctuation is a heading."""
    text = (line or "").strip()
    if not text or len(text) > config.HEADING_MAXIMUM_CHARACTERS:
        return False
    if len(text.split()) > config.HEADING_MAXIMUM_WORDS:
        return False
    if text[-1] in config.HEADING_TERMINAL_PUNCTUATION:
        return False
    if not text[0].isalpha() or not text[0].isupper():
        return False
    # Reject a fragment of wrapped prose that happens to start with a capital and
    # carries a mid-sentence comma; real headings in the policy book never do.
    return "," not in text


def _unwrap(lines):
    return re.sub(r"\s+", " ", " ".join(line.strip() for line in lines)).strip()


def split_into_sections(text):
    """Each section carries its heading and the body beneath it; nothing before it is lost."""
    lines = [line for line in (text or "").splitlines()]
    document_name = ""
    for line in lines:
        if line.strip():
            document_name = line.strip()
            break
    sections = []
    heading, body = document_name, []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if looks_like_heading(stripped) and stripped != document_name:
            if body or sections or heading != document_name:
                sections.append({"heading": heading, "body": _unwrap(body)})
            heading, body = stripped, []
        elif stripped == document_name and not body and not sections:
            continue
        else:
            body.append(stripped)
    sections.append({"heading": heading, "body": _unwrap(body)})
    return [section for section in sections if section["body"]]


def split_without_breaking_words(body, maximum_characters, overlap_characters):
    """Chunks of at most `maximum_characters`, cut at spaces, overlapping by about `overlap`."""
    text = re.sub(r"\s+", " ", body or "").strip()
    if len(text) <= maximum_characters:
        return [text] if text else []
    words = text.split(" ")
    chunks, start = [], 0
    while start < len(words):
        length, end = 0, start
        while end < len(words) and length + len(words[end]) + (1 if end > start else 0) \
                <= maximum_characters:
            length += len(words[end]) + (1 if end > start else 0)
            end += 1
        if end == start:  # a single word longer than the maximum
            end = start + 1
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        # Step back over whole words until roughly the overlap is repeated.
        back, carried = end, 0
        while back > start + 1 and carried < overlap_characters:
            back -= 1
            carried += len(words[back]) + 1
        start = back
    return chunks


def chunk_documents(documents, maximum_characters=None, overlap_characters=None):
    """Labelled chunks, CH####, unique and stable across all documents."""
    maximum = config.CHUNK_SIZE_CHARACTERS if maximum_characters is None else maximum_characters
    overlap = config.CHUNK_OVERLAP_CHARACTERS if overlap_characters is None else overlap_characters
    chunks = []
    for document in documents:
        for section in split_into_sections(document["text"]):
            for piece in split_without_breaking_words(section["body"], maximum, overlap):
                chunks.append({
                    "chunk_id": "CH{0:04d}".format(len(chunks) + 1),
                    "document_id": document["document_id"],
                    "document_title": document["title"],
                    "topic": document["topic"],
                    "heading": section["heading"],
                    "text": piece,
                })
    return chunks
