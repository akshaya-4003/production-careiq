"""One way to call a language model, whichever model is behind it.

Two backends share one interface. The offline backend is not a mock that returns
filler: it reads the facts the prompt was built from and answers in the format
the caller declared, so the whole workflow runs and its parsers are exercised
with no key and no network.
"""

import hashlib
import json
import os
import re
import time
from datetime import datetime, timezone

from config import config


class LanguageModelError(Exception):
    """The model could not be reached, or answered unusably twice running."""


# ---------------------------------------------------------------- parsing
_BULLET = re.compile(r"^\s*(?:[-*•#>]+|\d+[.)])\s*")


def _clean_label_line(line):
    return _BULLET.sub("", line).replace("**", "").strip()


def parse_labelled_response(response_text, field_labels):
    """Read LABEL: value lines into a dictionary; a missing label is an empty string."""
    wanted = {label.upper(): label for label in field_labels}
    parsed = {label: "" for label in field_labels}
    current = None
    for raw_line in (response_text or "").splitlines():
        line = _clean_label_line(raw_line)
        match = re.match(r"^([A-Za-z][A-Za-z /_-]{0,40}?)\s*:\s*(.*)$", line)
        if match and match.group(1).strip().upper() in wanted:
            current = wanted[match.group(1).strip().upper()]
            parsed[current] = match.group(2).strip()
            continue
        if match and match.group(1).strip().upper() not in wanted and \
                match.group(1).strip().isupper():
            current = None  # another, undeclared label
            continue
        if current and line:
            parsed[current] = (parsed[current] + " " + line).strip()
    return parsed


def parse_numbered_section(response_text, section_label, known_labels=None):
    """The numbered or bulleted lines under a section heading, up to the next label."""
    lines = (response_text or "").splitlines()
    stop_labels = set(label.upper() for label in (known_labels or []))
    collected, inside = [], False
    for raw_line in lines:
        line = raw_line.strip()
        cleaned = _clean_label_line(line)
        heading = re.match(r"^([A-Za-z][A-Za-z /_-]{0,40}?)\s*:\s*(.*)$", cleaned)
        if heading and heading.group(1).strip().upper() == section_label.upper():
            inside = True
            remainder = heading.group(2).strip()
            if remainder:
                collected.append(remainder)
            continue
        if inside and heading and (heading.group(1).strip().upper() in stop_labels
                                   or heading.group(1).strip().isupper()):
            break
        if inside and cleaned:
            collected.append(cleaned)
    return collected


def facts_from_prompt(prompt_text):
    """The indented 'Label: value' facts a prompt states, as a dictionary."""
    facts = {}
    for line in (prompt_text or "").splitlines():
        match = re.match(r"^\s{2,}([A-Za-z][^:]{0,60}):\s*(.*)$", line)
        if match:
            facts[match.group(1).strip()] = match.group(2).strip()
    return facts


def passages_from_prompt(prompt_text):
    """The labelled passages inside a prompt's delimited context block."""
    inside = re.search(r"<<<RETRIEVED POLICY PASSAGES>>>(.*?)<<<END OF PASSAGES>>>",
                       prompt_text or "", flags=re.DOTALL)
    if not inside:
        return []
    passages, current = [], None
    for line in inside.group(1).splitlines():
        header = re.match(r"^\[(CH\d{4})\]\s+(\S+)\s+-\s+(.*?)\s+/\s+(.*)$", line.strip())
        if header:
            current = {"chunk_id": header.group(1), "document_id": header.group(2),
                       "document_title": header.group(3), "heading": header.group(4),
                       "text": ""}
            passages.append(current)
        elif current is not None and line.strip():
            current["text"] = (current["text"] + " " + line.strip()).strip()
    return passages


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------- interface
class LLMClient(object):
    """generate(prompt_text, agent_name) -> (response_text, call_record)."""

    name = "base"
    model_name = ""

    def __init__(self):
        self.expected_labels = []
        self.expected_sections = []

    def expect(self, labels, sections=None):
        """Declare the reply shape. The real backend ignores it; the stand-in uses it."""
        self.expected_labels = list(labels or [])
        self.expected_sections = list(sections or [])

    def _complete(self, prompt_text, agent_name):
        raise NotImplementedError

    def generate(self, prompt_text, agent_name):
        started = time.time()
        response = self._complete(prompt_text, agent_name)
        record = {
            "agent_name": agent_name,
            "backend": self.name,
            "model_name": self.model_name,
            "prompt_characters": len(prompt_text or ""),
            "response_characters": len(response or ""),
            "latency_ms": int(round((time.time() - started) * 1000)),
            "created_at": _now(),
        }
        return response, record


# ---------------------------------------------------------------- offline backend
_WORDS = re.compile(r"[a-z0-9]+")


def _content_words(text):
    stop = set(config.OFFLINE_EMBEDDING_STOPWORDS)
    words = [config.QUESTION_SYNONYMS.get(word, word)
             for word in _WORDS.findall((text or "").lower()) if word not in stop]
    return [word[:-1] if word.endswith("s") and len(word) > 4 else word for word in words]


def _best_sentence(question, passages):
    """The passage sentence sharing most words with the question, and that share."""
    wanted = set(_content_words(question))
    best, best_share, best_label = "", 0.0, ""
    for passage in passages:
        for sentence in re.split(r"(?<=[.!?])\s+", passage["text"]):
            words = set(_content_words(sentence + " " + passage["heading"]))
            share = len(wanted & words) / float(len(wanted)) if wanted else 0.0
            if share > best_share:
                best, best_share, best_label = sentence.strip(), share, passage["chunk_id"]
    return best, best_share, best_label


class OfflineClient(LLMClient):
    """A deterministic stand-in: answers in the declared format from the prompt's own facts."""

    name = "offline"
    model_name = config.OFFLINE_MODEL_NAME

    def _complete(self, prompt_text, agent_name):
        facts = facts_from_prompt(prompt_text)
        passages = passages_from_prompt(prompt_text)
        answers = _offline_answers(agent_name, facts, passages)
        lines = []
        for label in self.expected_labels:
            lines.append(label + ": " + answers.get(label, facts.get(label.title(), "")))
        for section in self.expected_sections:
            lines.append(section + ":")
            for position, step in enumerate(answers.get(section, []) or [], start=1):
                lines.append("{0}. {1}".format(position, step))
        return "\n".join(lines)


def _labels_text(passages, how_many=2):
    return " ".join("[" + passage["chunk_id"] + "]" for passage in passages[:how_many])


def _offline_answers(agent_name, facts, passages):
    """The stand-in's reply for each agent, written only from the facts it was handed."""
    order = facts.get("Order", "this order")
    if agent_name == config.AGENT_NAME_TRIAGE:
        return {
            "PLAN": "Confirm the delivery position of {0}, score its escalation risk, read the "
                    "customer's message for the issue mode, then retrieve the policy that "
                    "governs the remedy.".format(order),
            "FOCUS": "Delivery position: {0}; hours since last scan: {1}.".format(
                facts.get("Delivery position", "unknown"),
                facts.get("Hours since last scan", "unknown")),
        }
    if agent_name == config.AGENT_NAME_RISK:
        return {
            "RISK SUMMARY": "The model scored {0} at an escalation probability of {1}, which the "
                            "written thresholds band as {2} and, for a {3} customer, priority "
                            "{4}.".format(order, facts.get("Escalation probability", ""),
                                          facts.get("Risk band", ""),
                                          facts.get("Customer tier", ""),
                                          facts.get("Priority", "")),
            "KEY DRIVERS": "Delivery position {0}; {1} hours since the last courier scan; "
                           "courier {2}.".format(facts.get("Delivery position", ""),
                                                 facts.get("Hours since last scan", ""),
                                                 facts.get("Courier", "")),
        }
    if agent_name == config.AGENT_NAME_ISSUE:
        return {
            "AGREES": "YES",
            "ISSUE MODE": facts.get("Classifier issue mode", ""),
            "REASONING": "The classifier read the message as {0} with confidence {1}, and the "
                         "message wording is consistent with that reading.".format(
                             facts.get("Classifier issue mode", ""),
                             facts.get("Classifier confidence", "")),
        }
    if agent_name == config.AGENT_NAME_POLICY:
        cited = _labels_text(passages)
        first = passages[0] if passages else {}
        return {
            "REMEDY": facts.get("Usual remedy for this issue", ""),
            "POLICY BASIS": "{0} {1}, section '{2}' {3}".format(
                first.get("document_id", ""), first.get("document_title", ""),
                first.get("heading", ""), cited).strip(),
            "CITATIONS": cited,
        }
    if agent_name == config.AGENT_NAME_COMPOSER:
        amount = facts.get("Cash committed (INR)", "0")
        remedy = facts.get("Remedy", "")
        money = "" if amount in ("0", "0.0", "0.00") else \
            " This commits INR {0} to you.".format(amount)
        return {
            "CUSTOMER REPLY": "We are sorry about the problem with order {0}. We will resolve "
                              "it by {1}.{2} We expect this to be completed within about {3} "
                              "hours.".format(order, remedy.lower(), money,
                                              facts.get("Estimated resolution hours", "")),
            "ACTION STEPS": [
                "Confirm the issue ({0}) on the ticket.".format(facts.get("Issue mode", "")),
                "Carry out the remedy: {0}.".format(remedy),
                "Record the policy basis: {0}.".format(facts.get("Policy basis", "")),
                "Update the customer with the outcome.",
            ],
        }
    if agent_name == config.AGENT_NAME_FINALIZER:
        decision = facts.get("Decision", "")
        remedy = facts.get("Final remedy", "")
        changed = []
        if facts.get("Draft remedy", "") != remedy:
            changed.append("remedy changed from {0} to {1}".format(facts.get("Draft remedy", ""),
                                                                  remedy))
        if facts.get("Manager feedback", ""):
            changed.append("manager feedback applied")
        return {
            "FINAL REPLY": "We are sorry about the problem with order {0}. We will resolve it "
                           "by {1}. {2}".format(order, remedy.lower(),
                                                facts.get("Manager feedback", "")).strip(),
            "WHAT CHANGED": "; ".join(changed) if changed else
                            "No change; the draft was {0}d as written.".format(decision.lower()),
            "ACTION STEPS": [
                "Confirm the issue ({0}) on the ticket.".format(facts.get("Issue mode", "")),
                "Carry out the remedy: {0}.".format(remedy),
                "Record the policy basis: {0}.".format(facts.get("Policy basis", "")),
                "Update the customer with the outcome.",
            ],
        }
    if agent_name == config.AGENT_NAME_ROUTER:
        suggestion = facts.get("Rule suggestion", config.INTENT_POLICY_QUESTION)
        return {"INTENT": suggestion or config.INTENT_POLICY_QUESTION,
                "REASON": "Followed the rule layer's suggestion."}
    if agent_name == config.AGENT_NAME_CHAT:
        sentence, share, label = _best_sentence(facts.get("Question", ""), passages)
        if not sentence:
            return {"ANSWER": "", "CONFIDENCE": "0.0", "CITATIONS": ""}
        # The stand-in's confidence is the coverage figure the prompt states as a fact.
        coverage = facts.get("Passage coverage of the question", "{0:.2f}".format(share))
        return {"ANSWER": sentence + " [" + label + "]",
                "CONFIDENCE": coverage,
                "CITATIONS": "[" + label + "]"}
    return {}


# ---------------------------------------------------------------- gemini backend
def gemini_is_available():
    """Whether the real backend can be used, and why not when it cannot."""
    if not os.environ.get(config.GEMINI_API_KEY_VARIABLE, "").strip():
        return False, config.GEMINI_API_KEY_VARIABLE + " is not set"
    try:
        from google import genai  # noqa: F401
    except ImportError:
        return False, "the google-genai library is not installed"
    return True, "key present and library installed"


class GeminiClient(LLMClient):
    """The real model through the free tier, with a prompt cache and rate-limit retries."""

    name = "gemini"

    def __init__(self):
        LLMClient.__init__(self)
        from google import genai

        self.model_name = config.GEMINI_MODEL_NAME
        self._client = genai.Client(api_key=os.environ.get(config.GEMINI_API_KEY_VARIABLE))
        self._cache = self._load_cache()

    @staticmethod
    def _load_cache():
        if os.path.exists(config.LLM_CACHE_FILE):
            try:
                with open(config.LLM_CACHE_FILE, "r", encoding="utf-8") as handle:
                    return json.load(handle)
            except (ValueError, OSError):
                return {}
        return {}

    def _save_cache(self):
        folder = os.path.dirname(config.LLM_CACHE_FILE)
        if folder:
            os.makedirs(folder, exist_ok=True)
        with open(config.LLM_CACHE_FILE, "w", encoding="utf-8") as handle:
            json.dump(self._cache, handle)

    def _complete(self, prompt_text, agent_name):
        from google.genai import types

        key = hashlib.sha256((self.model_name + "\n" + prompt_text).encode("utf-8")).hexdigest()
        if key in self._cache:
            return self._cache[key]
        last_error = None
        for attempt in range(config.GEMINI_RETRY_ATTEMPTS):
            try:
                reply = self._client.models.generate_content(
                    model=self.model_name, contents=prompt_text,
                    config=types.GenerateContentConfig(
                        max_output_tokens=config.LLM_MAX_OUTPUT_TOKENS,
                        temperature=config.LLM_TEMPERATURE))
                text = (reply.text or "").strip()
                if text:
                    self._cache[key] = text
                    self._save_cache()
                    return text
                last_error = "an empty reply"
            except Exception as error:  # the SDK raises several error types
                last_error = str(error)
                rate_limited = "429" in last_error or "RESOURCE_EXHAUSTED" in last_error
                if rate_limited and attempt < config.GEMINI_RETRY_ATTEMPTS - 1:
                    time.sleep(config.FREE_TIER_WAIT_SECONDS)
                    continue
                if not rate_limited:
                    break
        raise LanguageModelError("Gemini did not answer usably for " + agent_name + ": "
                                 + str(last_error))


def load_llm_client(backend=None):
    """'offline' insists on the stand-in, 'gemini' on the real model, 'auto' prefers the real one."""
    choice = (config.LLM_BACKEND if backend is None else backend).strip().lower()
    if choice == "offline":
        return OfflineClient()
    if choice == "gemini":
        available, reason = gemini_is_available()
        if not available:
            raise LanguageModelError("The gemini backend was requested and is unavailable: "
                                     + reason)
        return GeminiClient()
    if choice == "auto":
        available, reason = gemini_is_available()
        if available:
            try:
                client = GeminiClient()
                print("Language model backend: gemini")
                return client
            except Exception as error:
                reason = str(error)
        print("Language model backend: offline (" + reason + ")")
        return OfflineClient()
    raise ValueError("Unknown language model backend " + repr(choice)
                     + "; use 'gemini', 'offline' or 'auto'.")


def ask_for_labels(client, prompt_text, agent_name, labels, sections=None):
    """Declare the shape, call once, parse, and return everything an auditor needs."""
    client.expect(labels, sections)
    response, record = client.generate(prompt_text, agent_name)
    known = list(labels) + list(sections or [])
    parsed = parse_labelled_response(response, labels)
    parsed_sections = {section: parse_numbered_section(response, section, known)
                       for section in (sections or [])}
    return {"labels": parsed, "sections": parsed_sections, "call_record": record,
            "raw_response": response}
