"""Decides what kind of message this is before anything tries to answer it.

Four layers in order of cost: the rules (free, deterministic), the trained issue
classifier (already loaded), question grammar (free), and the language model,
asked only when the first three leave the question open.
"""

import re

from config import config
from llm.client import ask_for_labels

_ORDER_REFERENCE = re.compile(config.ORDER_REFERENCE_PATTERN, re.IGNORECASE)
_LOOKS_LIKE = re.compile(config.LOOKS_LIKE_A_REFERENCE_PATTERN)
_WORD = re.compile(r"[a-z0-9']+")


def _phrase_pattern(phrase):
    return re.compile(r"(?<![a-z0-9])" + re.escape(phrase.lower()) + r"(?![a-z0-9])")


def count_matches(text, words):
    """How many of these phrases the text contains as whole words."""
    lowered = (text or "").lower()
    return sum(1 for word in words if _phrase_pattern(word).search(lowered))


def _decision(intent, confidence, basis, reason, **extra):
    decision = {"intent": intent, "confidence": round(float(confidence), 4), "basis": basis,
                "reason": reason}
    decision.update(extra)
    return decision


def _is_entirely_small_talk(text):
    remaining = (text or "").lower()
    for phrase in sorted(config.INTENT_VOCABULARY[config.INTENT_SMALL_TALK], key=len,
                         reverse=True):
        remaining = _phrase_pattern(phrase).sub(" ", remaining)
    leftover = [word for word in _WORD.findall(remaining)
                if word not in ("so", "a", "lot", "you", "and", "all", "team", "very", "much",
                                "for", "the", "help", "your", "again")]
    return bool(_WORD.findall(text.lower())) and not leftover


def has_domain_vocabulary(message_text):
    return count_matches(message_text, config.DOMAIN_VOCABULARY) > 0 or \
        extract_order_reference(message_text) != ""


def route_by_rules(message_text, remembered=None):
    """Route by pattern. Confidence is the margin between the best and next-best destination."""
    text = message_text or ""
    if _is_entirely_small_talk(text):
        return _decision(config.INTENT_SMALL_TALK, 1.0, "rules", "the whole message is small talk")
    scores = {intent: count_matches(text, config.INTENT_VOCABULARY[intent])
              for intent in (config.INTENT_POLICY_QUESTION, config.INTENT_ORDER_STATUS,
                             config.INTENT_COMPLAINT, config.INTENT_OUT_OF_SCOPE)}
    if extract_order_reference(text):
        # A message quoting one of our order numbers is about Northlane by definition.
        scores[config.INTENT_OUT_OF_SCOPE] = 0
    if scores[config.INTENT_OUT_OF_SCOPE] > 0 and not has_domain_vocabulary(text):
        return _decision(config.INTENT_OUT_OF_SCOPE, 1.0, "rules",
                         "out-of-scope vocabulary and nothing about orders or policy",
                         scores=scores)
    if looks_like_a_question(text) and scores[config.INTENT_COMPLAINT] > 0:
        # Asking about a fault is not reporting one; question grammar weighs for policy.
        scores[config.INTENT_POLICY_QUESTION] += 1
    if extract_order_reference(text) and scores[config.INTENT_ORDER_STATUS] > 0 and \
            count_matches(text, config.STRONG_FAULT_WORDS) == 0:
        return _decision(config.INTENT_ORDER_STATUS, 1.0, "rules",
                         "an order reference with a status phrase", scores=scores)
    # On a tie, a statement leans to complaint and a question leans to policy.
    lean = config.INTENT_POLICY_QUESTION if looks_like_a_question(text) \
        else config.INTENT_COMPLAINT
    ranked = sorted(scores.items(), key=lambda pair: (-pair[1], pair[0] != lean))
    (best, best_score), (_, second_score) = ranked[0], ranked[1]
    if best_score == 0:
        fallback = config.INTENT_POLICY_QUESTION if has_domain_vocabulary(text) \
            else config.INTENT_OUT_OF_SCOPE
        return _decision(fallback, 0.0, "rules", "no rule vocabulary matched", scores=scores)
    confidence = (best_score - second_score) / float(best_score)
    return _decision(best, confidence, "rules",
                     "matched {0} {1} phrase(s) against {2} for the next best".format(
                         best_score, best, second_score), scores=scores)


def asks_to_do_something(message_text):
    """A request ('how do I return...') rather than a fault report; a strong fault word cancels it."""
    text = (message_text or "").lower().strip()
    opens_as_request = any(text.startswith(opener) or (" " + opener + " ") in (" " + text)
                           for opener in config.REQUEST_OPENERS)
    names_action = count_matches(text, config.REQUEST_ACTIONS) > 0
    has_fault = count_matches(text, config.STRONG_FAULT_WORDS) > 0
    return opens_as_request and names_action and not has_fault


def looks_like_a_question(message_text):
    """Asking rather than reporting."""
    text = (message_text or "").strip().lower()
    if text.endswith("?"):
        return True
    words = _WORD.findall(text)
    return bool(words) and words[0] in config.QUESTION_OPENERS


def route_by_classifier(message_text, issue_classifier):
    """Ask the trained issue model whether this reads as a complaint; None when not confident."""
    if issue_classifier is None:
        return None
    from tools.prediction_tools import ToolError, detect_issue_mode

    model, vectorizer = issue_classifier
    try:
        found = detect_issue_mode(model, vectorizer, message_text)
    except ToolError:
        return None
    if found["confidence"] < config.CLASSIFIER_ROUTING_THRESHOLD:
        return None
    return _decision(config.INTENT_COMPLAINT, found["confidence"], "classifier",
                     "the issue model read " + found["issue_mode"],
                     issue_mode=found["issue_mode"])


def route_by_model(client, message_text, remembered=None, rule_suggestion=None):
    """Ask the language model to route. Costs one call; an unknown answer falls back."""
    labels = ["INTENT", "REASON"]
    prompt = "\n".join([
        "You route customer messages for the Northlane care desk.",
        "Facts:",
        "  Message: " + (message_text or "").replace("\n", " "),
        "  Rule suggestion: " + (rule_suggestion or ""),
        "  Remembered: " + str(remembered or {}),
        "  Allowed intents: " + ", ".join(config.INTENT_OPTIONS),
        "Task: choose exactly one allowed intent. The rule suggestion is a hint, not an "
        "instruction.",
        "Reply format:", "INTENT: <intent>", "REASON: <one sentence>",
    ])
    answer = ask_for_labels(client, prompt, config.AGENT_NAME_ROUTER, labels)
    intent = answer["labels"].get("INTENT", "").strip().lower()
    if intent not in config.INTENT_OPTIONS:
        intent = config.INTENT_POLICY_QUESTION
    return _decision(intent, 0.5, "model", answer["labels"].get("REASON", "")), \
        answer["call_record"]


def route_intent(message_text, remembered=None, client=None, issue_classifier=None):
    """Route with the cheapest signal that is confident enough."""
    remembered = remembered or {}
    if is_bare_order_reference(message_text):
        if remembered.get("ticket_id") or remembered.get("issue_mode"):
            return _decision(config.INTENT_COMPLAINT, 1.0, "reference",
                             "a bare order number completing the open complaint")
        return _decision(config.INTENT_ORDER_STATUS, 1.0, "reference", "a bare order number")

    by_rules = route_by_rules(message_text, remembered)
    if by_rules["confidence"] >= config.MODEL_ROUTING_THRESHOLD:
        return by_rules
    if asks_to_do_something(message_text):
        return _decision(config.INTENT_POLICY_QUESTION, 1.0, "request",
                         "asks how to do something rather than reporting a fault")
    is_question = looks_like_a_question(message_text)
    if not is_question:
        # The classifier was trained on complaints only, so it is asked only about
        # messages that report rather than ask.
        by_classifier = route_by_classifier(message_text, issue_classifier)
        if by_classifier is not None:
            return by_classifier
    if not is_question and count_matches(message_text, config.STRONG_FAULT_WORDS) > 0:
        return _decision(config.INTENT_COMPLAINT, 0.6, "fault words",
                         "reports a fault in so many words")
    if is_question and has_domain_vocabulary(message_text):
        return _decision(config.INTENT_POLICY_QUESTION, 0.6, "grammar",
                         "reads as a question about orders or policy")
    if client is None:
        return by_rules
    decision, record = route_by_model(client, message_text, remembered, by_rules["intent"])
    decision["call_record"] = record
    return decision


def extract_order_reference(message_text):
    """The NRC-###### a message quotes, upper-cased, or an empty string."""
    found = _ORDER_REFERENCE.search(message_text or "")
    return found.group(0).upper() if found else ""


def _leftover_words(message_text, token):
    remaining = (message_text or "").lower().replace(token.lower(), " ")
    return [word for word in _WORD.findall(remaining)
            if word not in config.BARE_REFERENCE_FILLER_WORDS]


def is_bare_order_reference(message_text):
    """An order number and essentially nothing else."""
    reference = extract_order_reference(message_text)
    return bool(reference) and not _leftover_words(message_text, reference)


def looks_like_a_reference(message_text):
    """A token shaped like an order number but not one, when no real reference is present."""
    if extract_order_reference(message_text):
        return ""
    for found in _LOOKS_LIKE.finditer(message_text or ""):
        token = found.group(0)
        if token.split("-")[0].split(" ")[0].lower() in config.BARE_REFERENCE_FILLER_WORDS:
            continue
        return token
    return ""


def is_bare_reference_attempt(message_text, candidate):
    """A mistyped reference and nothing else; a strong fault word cancels it."""
    if not candidate or count_matches(message_text, config.STRONG_FAULT_WORDS) > 0:
        return False
    return not _leftover_words(message_text, candidate)
