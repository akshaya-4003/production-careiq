"""Searches the policy book and returns the text exactly as written.

These are tools, not agents. The policy advisor reads what comes back, and
because it only ever sees retrieved text, every instruction can be traced.
"""

import re

from pydantic import BaseModel, Field, ValidationError

from config import config
from knowledge.procedure_retrieval import (
    cited_documents,
    format_retrieved_context,
    open_policy_knowledge,
)
from tools.prediction_tools import ToolError

_CITATION = re.compile(r"\[(CH\d{4})\]")


class PolicyRequest(BaseModel):
    question: str = Field(min_length=config.MINIMUM_MESSAGE_CHARACTERS,
                          max_length=config.MAXIMUM_MESSAGE_CHARACTERS)


def expand_issue_query(issue_mode, product_category):
    """A query rich in the words the governing passage is likely to use."""
    hints = config.ISSUE_QUERY_HINTS.get(issue_mode, "")
    return " ".join(part for part in (issue_mode, product_category, hints) if part).strip()


def open_knowledge(store_path=None, collection_name=None, embedder=None):
    """The policy index paired with the embedder that can query it."""
    return open_policy_knowledge(store_path, collection_name, embedder)


def search_policy_documents(knowledge, question, how_many=None, use_topic_filter=True):
    """Passages, prompt context, source documents and chunk labels for one question."""
    try:
        request = PolicyRequest(question=str(question or "").strip())
    except ValidationError as error:
        raise ToolError("PolicyRequest refused: " + "; ".join(item["msg"]
                                                             for item in error.errors()))
    passages = knowledge.search(request.question, how_many, use_topic_filter)
    if not passages:
        raise ToolError("Nothing in the policy book matched the question " + repr(question))
    return {
        "question": request.question,
        "passages": passages,
        "context_text": format_retrieved_context(passages),
        "source_documents": cited_documents(passages),
        "chunk_ids": [passage["chunk_id"] for passage in passages],
    }


def lookup_remedy_policy(knowledge, issue_mode, product_category, how_many=None):
    """What the book says about remedying this issue. The topic filter is deliberately off."""
    return search_policy_documents(knowledge, expand_issue_query(issue_mode, product_category),
                                   how_many, use_topic_filter=False)


def lookup_authority_policy(knowledge, remedy_type, how_many=None):
    """What the book says about who may authorise this remedy."""
    return search_policy_documents(knowledge, config.AUTHORITY_QUERY_TEMPLATE.format(remedy_type),
                                   how_many, use_topic_filter=False)


def verify_citations(answer_text, retrieved_chunks):
    """Labels the answer cites, and the ones it cites that it was never handed."""
    handed = set(chunk["chunk_id"] for chunk in retrieved_chunks)
    cited, invented = [], []
    for label in _CITATION.findall(answer_text or ""):
        target = cited if label in handed else invented
        if label not in target:
            target.append(label)
    return cited, invented
