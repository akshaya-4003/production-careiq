"""Short-term memory scoped to one conversation, and nothing wider."""

from datetime import datetime, timedelta

from config import config
from persistence import repository


def start_session(session, session_id, user_name=""):
    """Open a conversation, or pick up one already open."""
    return repository.get_or_create_chat_session(session, session_id, user_name)


def remember(session, session_id, updates):
    """Merge settled facts. Only allowed keys are kept; an empty value never erases a known one."""
    known = repository.read_remembered_entities(session, session_id)
    for key in config.MEMORY_ENTITY_KEYS:
        value = (updates or {}).get(key)
        if value not in (None, ""):
            known[key] = value
    known = {key: known[key] for key in config.MEMORY_ENTITY_KEYS if key in known}
    repository.write_remembered_entities(session, session_id, known)
    return known


def recall(session, session_id):
    """What this conversation has settled so far."""
    known = repository.read_remembered_entities(session, session_id)
    return {key: known[key] for key in config.MEMORY_ENTITY_KEYS if key in known}


def record_turn(session, session_id, speaker, content, intent="", citations="", refused_reason=""):
    """Write one turn of the transcript, with what it was routed as and what it stood on."""
    if isinstance(citations, (list, tuple)):
        citations = ",".join(citations)
    return repository.append_chat_turn(session, {
        "session_id": session_id, "speaker": speaker, "content": content or "",
        "intent": intent or "", "citations": citations or "",
        "refused_reason": refused_reason or ""})


def conversation_so_far(session, session_id, how_many=None):
    """The last few turns, oldest first."""
    count = config.MEMORY_MAX_TURNS if how_many is None else how_many
    return repository.recent_chat_turns(session, session_id, count)


def format_history(turns):
    if not turns:
        return "This is the first message of the conversation."
    return "\n".join("{0}: {1}".format(turn["speaker"], turn["content"]) for turn in turns)


def format_entities(entities):
    known = [(key, entities[key]) for key in config.MEMORY_ENTITY_KEYS
             if entities.get(key) not in (None, "")]
    if not known:
        return "Nothing has been settled yet."
    return "\n".join("{0}: {1}".format(key, value) for key, value in known)


def resolve_order_reference(explicit_reference, remembered):
    """The order this turn is about, and whether it was named now or carried from earlier."""
    if explicit_reference:
        return explicit_reference, "named in this message"
    carried = (remembered or {}).get("order_id", "")
    if carried:
        return carried, "carried from earlier in the conversation"
    return "", "no order named or remembered"


def session_has_gone_idle(session_record, minutes=None):
    """Whether a conversation has been quiet long enough to be finished with."""
    limit = config.CHAT_SESSION_IDLE_MINUTES if minutes is None else minutes
    last = (session_record or {}).get("last_active_at")
    if not last:
        return True
    try:
        moment = datetime.fromisoformat(str(last))
    except ValueError:
        return True
    return datetime.utcnow() - moment.replace(tzinfo=None) > timedelta(minutes=limit)
