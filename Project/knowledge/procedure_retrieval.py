"""Retail-specific retrieval: topics, prompt context, and which documents an answer cites."""

import re

from config import config
from knowledge.embedding import embedder_for_backend_name
from knowledge.vector_index import index_backend_name, open_policy_index, search_policy


def _contains_phrase(text, phrase):
    return re.search(r"\b" + re.escape(phrase) + r"\b", text) is not None


def topics_in_question(question):
    """The configured topics whose vocabulary the question uses, in config order."""
    text = (question or "").lower()
    return [topic for topic, words in config.TOPIC_VOCABULARY.items()
            if any(_contains_phrase(text, word) for word in words)]


def topic_filter(topics):
    """The store's metadata filter for these topics, or None for an empty list."""
    if not topics:
        return None
    if len(topics) == 1:
        return {"topic": topics[0]}
    return {"topic": {"$in": list(topics)}}


def retrieve_relevant_chunks(collection, question, embedder, how_many=None, use_topic_filter=True):
    """Search by topic when it helps; fall back to the whole book when a filter finds nothing."""
    where = topic_filter(topics_in_question(question)) if use_topic_filter else None
    found = search_policy(collection, question, embedder, how_many, where)
    if where is not None and not found:
        found = search_policy(collection, question, embedder, how_many, None)
    return found


def format_retrieved_context(chunks):
    """Delimited passages, each with its label, for a prompt."""
    lines = ["<<<RETRIEVED POLICY PASSAGES>>>"]
    for chunk in chunks:
        lines.append("[{0}] {1} - {2} / {3}".format(
            chunk["chunk_id"], chunk["document_id"], chunk["document_title"], chunk["heading"]))
        lines.append(chunk["text"])
        lines.append("")
    lines.append("<<<END OF PASSAGES>>>")
    return "\n".join(lines)


def cited_documents(chunks):
    """The distinct document codes a set of passages came from, in first-seen order."""
    seen = []
    for chunk in chunks:
        if chunk["document_id"] not in seen:
            seen.append(chunk["document_id"])
    return seen


class PolicyKnowledge(object):
    """The collection and the embedder that built it, held together."""

    def __init__(self, collection, embedder):
        self.collection = collection
        self.embedder = embedder

    def search(self, question, how_many=None, use_topic_filter=True):
        return retrieve_relevant_chunks(self.collection, question, self.embedder, how_many,
                                        use_topic_filter)

    def count(self):
        return int(self.collection.count())

    def backend_name(self):
        return self.embedder.name


def open_policy_knowledge(store_path=None, collection_name=None, embedder=None):
    """Open the index paired with an embedder that can query it."""
    collection = open_policy_index(store_path, collection_name)
    recorded = index_backend_name(collection)
    if embedder is None:
        embedder = embedder_for_backend_name(recorded)
    elif recorded not in ("unknown", embedder.name):
        raise ValueError("The policy index was built with " + recorded + " and is being queried "
                         "with " + embedder.name + ". Rebuild it with "
                         "python3 run_knowledge_indexing.py.")
    return PolicyKnowledge(collection, embedder)
