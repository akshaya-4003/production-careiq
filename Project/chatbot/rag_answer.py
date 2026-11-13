"""Answers a policy question from the retrieved passages and from nothing else."""

import re

from config import config
from llm.client import ask_for_labels
from tools.prediction_tools import ToolError
from tools.retrieval_tools import search_policy_documents, verify_citations

_WORD = re.compile(r"[a-z0-9]+")


def _subject_words(text):
    ignored = set(config.OFFLINE_EMBEDDING_STOPWORDS) | set(config.OFFLINE_CONFIDENCE_IGNORED_WORDS)
    words = []
    for word in _WORD.findall((text or "").lower()):
        word = config.QUESTION_SYNONYMS.get(word, word)
        if word not in ignored and len(word) > 2:
            words.append(word[:-1] if word.endswith("s") and len(word) > 4 else word)
    return words


def extractive_answer(question, passages):
    """A no-call answer: the passage sentence that covers most of the question, with its label.

    Confidence is the share of the question's subject words the chosen passage
    covers, so a passage that merely sits nearest in the index scores low.
    """
    wanted = set(_subject_words(question))
    if not wanted:
        return "", 0.0, ""
    best = ("", 0.0, "")
    for passage in passages:
        covered = wanted & set(_subject_words(passage["heading"] + " " + passage["text"]))
        share = len(covered) / float(len(wanted))
        sentences = re.split(r"(?<=[.!?])\s+", passage["text"])
        sentence = max(sentences, key=lambda s: len(wanted & set(_subject_words(s))))
        if share > best[1]:
            best = (sentence.strip(), share, passage["chunk_id"])
    return best


def _refusal(reason, found=None, record=None):
    return {"answer": config.LOW_CONFIDENCE_REFUSAL, "citations": [],
            "source_documents": [], "refused_reason": reason, "confidence": 0.0,
            "call_record": record, "passages": (found or {}).get("passages", [])}


def answer_or_refuse(client, knowledge, question, history_text, entities_text):
    """Retrieve, answer with checked citations, and refuse rather than guess.

    One call is spent, or none when retrieval settles that there is nothing to
    answer from, or when no client is supplied (the extractive path).
    """
    try:
        found = search_policy_documents(knowledge, question)
    except ToolError as error:
        return _refusal("retrieval found nothing to answer from: " + str(error))

    if client is None:
        sentence, confidence, label = extractive_answer(question, found["passages"])
        if not sentence or confidence < config.MINIMUM_ANSWER_CONFIDENCE:
            return _refusal("the best passage covers only {0:.0%} of the question".format(
                confidence), found)
        answer_text, record = sentence + " [" + label + "]", None
        check_text = answer_text
    else:
        _, coverage, _ = extractive_answer(question, found["passages"])
        prompt = "\n\n".join([
            "You answer Northlane customers using only the retrieved policy passages.",
            "Facts:\n  Question: " + question.replace("\n", " ")
            + "\n  Passage coverage of the question: {0:.2f}".format(coverage),
            "Conversation so far:\n" + history_text,
            "Settled in this conversation:\n" + entities_text,
            found["context_text"],
            "Task: answer in one to three sentences using only the passages. Put the label of "
            "every passage you rely on in square brackets after the claim, for example "
            "[CH0001]. If the passages do not answer the question, say so and give a "
            "confidence below 0.5.",
            "Reply format:\nANSWER: <answer>\nCONFIDENCE: <0.0 to 1.0>\nCITATIONS: <labels>",
        ])
        reply = ask_for_labels(client, prompt, config.AGENT_NAME_CHAT,
                               ["ANSWER", "CONFIDENCE", "CITATIONS"])
        record = reply["call_record"]
        answer_text = reply["labels"].get("ANSWER", "").strip()
        try:
            confidence = float(re.findall(r"[0-9.]+", reply["labels"].get("CONFIDENCE", "0"))[0])
        except (IndexError, ValueError):
            confidence = 0.0
        cited_text = answer_text + " " + reply["labels"].get("CITATIONS", "")
        if not answer_text or confidence < config.MINIMUM_ANSWER_CONFIDENCE:
            return _refusal("answer confidence {0:.2f} is below {1}".format(
                confidence, config.MINIMUM_ANSWER_CONFIDENCE), found, record)
        check_text = cited_text

    cited, invented = verify_citations(check_text, found["passages"])
    if invented:
        return _refusal("the answer cited passages it was never handed: "
                        + ", ".join(invented), found, record)
    if not cited:
        return _refusal("the answer cited no passage", found, record)
    sources = []
    for passage in found["passages"]:
        if passage["chunk_id"] in cited and passage["document_id"] not in sources:
            sources.append(passage["document_id"])
    return {"answer": answer_text, "citations": cited, "source_documents": sources,
            "refused_reason": "", "confidence": round(float(confidence), 4),
            "call_record": record, "passages": found["passages"]}
