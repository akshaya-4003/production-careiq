"""Turns a customer's message into the words that carry the complaint.

Every removal describes how the message was sent (markup, greeting, sign-off,
signature, mail footer, quoted reply, order number) rather than what went wrong.
"""

import html
import re

import pandas as pd

from config import config
from data_pipeline import common_cleaning as steps

_MARKUP = re.compile(config.MARKUP_PATTERN)
_FOOTERS = [re.compile(pattern, re.IGNORECASE) for pattern in config.MAIL_FOOTER_PATTERNS]
_GREETINGS = [re.compile(pattern, re.IGNORECASE) for pattern in config.GREETING_PATTERNS]
_SIGN_OFFS = [re.compile(pattern, re.IGNORECASE) for pattern in config.SIGN_OFF_PATTERNS]
_SIGNATURE_LINES = [re.compile(pattern, re.IGNORECASE)
                    for pattern in config.SIGNATURE_LINE_PATTERNS]
_SIGNATURE_STARTS = [re.compile(pattern, re.IGNORECASE)
                     for pattern in config.SIGNATURE_START_PATTERNS]
_QUOTED = [re.compile(pattern, re.IGNORECASE) for pattern in config.QUOTED_REPLY_PATTERNS]
_ORDER_REFERENCE = re.compile(r"(?:\bref(?:erence)?\.?:?\s*)?" + config.ORDER_REFERENCE_PATTERN,
                              re.IGNORECASE)


def _repair_encoding(text):
    for damaged, repaired in config.ENCODING_REPAIR_MAP.items():
        text = text.replace(damaged, repaired)
    text = html.unescape(text)
    # Drop the replacement character and other unprintable debris.
    return "".join(character for character in text
                   if character.isprintable() or character in "\n\t")


def clean_message_text(message_text):
    """Reduce one message to lower-cased, single-spaced complaint words."""
    if message_text is None or (not isinstance(message_text, str) and pd.isna(message_text)):
        return ""
    text = _repair_encoding(str(message_text))
    text = _MARKUP.sub("\n", text)
    for footer in _FOOTERS:
        text = footer.sub(" ", text)

    kept_lines = []
    for line in text.splitlines():
        if any(pattern.search(line) for pattern in _QUOTED):
            # Everything from a quoted earlier reply onwards belongs to that reply.
            break
        if kept_lines and any(pattern.match(line) for pattern in _SIGNATURE_STARTS):
            # The signature block: it describes the sender, not the problem.
            break
        if any(pattern.match(line) for pattern in _SIGNATURE_LINES):
            continue
        kept_lines.append(line)
    text = " ".join(kept_lines)

    text = _ORDER_REFERENCE.sub(" " + config.ORDER_REFERENCE_MASK + " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    for greeting in _GREETINGS:
        text = greeting.sub("", text, count=1).strip()
    # A sign-off can sit before the masked reference, so strip the mask first and
    # put it back after the sign-off is gone.
    text = text.rstrip(" .,;")
    had_reference = text.lower().endswith(config.ORDER_REFERENCE_MASK)
    if had_reference:
        text = text[: -len(config.ORDER_REFERENCE_MASK)].rstrip(" .,")
    for _ in range(2):
        for sign_off in _SIGN_OFFS:
            text = sign_off.sub("", text).strip()
    if had_reference:
        text = text + " " + config.ORDER_REFERENCE_MASK
    text = text.lower()
    text = re.sub(r"[^a-z0-9'%\-\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def is_usable_message(cleaned_text):
    """True when a cleaned message is long enough and has enough words to classify."""
    if not cleaned_text or not isinstance(cleaned_text, str):
        return False
    text = cleaned_text.strip()
    if len(text) < config.MINIMUM_MESSAGE_CHARACTERS:
        return False
    words = [word for word in text.split() if word != config.ORDER_REFERENCE_MASK
             and any(character.isalpha() for character in word)]
    return len(words) >= config.MINIMUM_MESSAGE_WORDS


def clean_message_frame(frame):
    """Clean every message, then drop unusable rows, labels outside the catalogue and duplicates."""
    cleaned = steps.normalise_missing(frame, config.NULL_TOKEN_LIST)
    text_column = config.MESSAGE_TEXT_COLUMN
    label_column = config.MESSAGE_TARGET_COLUMN
    cleaned[text_column] = cleaned[text_column].map(clean_message_text)
    cleaned = cleaned[cleaned[text_column].map(is_usable_message)]
    cleaned = steps.apply_category_mapping(cleaned, label_column, config.ISSUE_MODE_MAPPING,
                                           config.UNKNOWN_CATEGORY_VALUE)
    cleaned = cleaned[cleaned[label_column].isin(config.VALID_ISSUE_MODES)]
    cleaned = steps.apply_category_mapping(cleaned, "channel", config.CHANNEL_MAPPING,
                                           config.UNKNOWN_CATEGORY_VALUE)
    other_columns = [column for column in cleaned.columns if column != text_column]
    cleaned = steps.fill_missing_categorical(cleaned, other_columns,
                                             config.UNKNOWN_CATEGORY_VALUE)
    cleaned = steps.drop_duplicate_rows(cleaned, [config.MESSAGE_IDENTIFIER_COLUMN])
    return cleaned.reset_index(drop=True)
