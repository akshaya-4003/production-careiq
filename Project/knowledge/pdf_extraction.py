"""Reads the six policy documents and attaches the identity the platform knows each by."""

import os

from pypdf import PdfReader

from config import config


def read_document_text(pdf_path):
    """Every page's text in page order, as one string. An empty extraction raises."""
    reader = PdfReader(pdf_path)
    pages = [(page.extract_text() or "").strip() for page in reader.pages]
    text = "\n".join(page for page in pages if page)
    if not text.strip():
        raise ValueError("No text could be extracted from " + str(pdf_path)
                         + "; it would be a silent hole in the policy index.")
    return text


def _document_entry(document_id):
    for entry in config.POLICY_DOCUMENTS:
        if entry["document_id"] == document_id:
            return entry
    return {}


def document_title_for(document_id):
    """The title config gives a document code, or an empty string for an unknown code."""
    return _document_entry(document_id).get("title", "")


def document_topic_for(document_id):
    """The retrieval topic config gives a document code, or an empty string."""
    return _document_entry(document_id).get("topic", "")


def load_policy_documents(folder_path=None):
    """Load every configured document with its id, title, topic and text."""
    folder = config.DOCS_DIRECTORY if folder_path is None else folder_path
    missing = []
    documents = []
    for entry in config.POLICY_DOCUMENTS:
        path = os.path.join(folder, entry["document_id"] + ".pdf")
        if not os.path.exists(path):
            missing.append(entry["document_id"])
            continue
        documents.append({
            "document_id": entry["document_id"],
            "title": document_title_for(entry["document_id"]),
            "topic": document_topic_for(entry["document_id"]),
            "path": path,
            "text": read_document_text(path),
        })
    if missing:
        raise FileNotFoundError(
            "These policy documents are listed in config but missing from " + folder
            + ": " + ", ".join(missing))
    return documents
