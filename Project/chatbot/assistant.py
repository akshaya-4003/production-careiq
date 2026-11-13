"""The chatbot's one entry point: route, answer from the right source, remember, record.

At most one language model call per turn, and often none. A real complaint opens
a ticket and leaves the resolution to the advisory workflow.
"""

import uuid

from chatbot import memory
from chatbot.intent_router import (
    extract_order_reference,
    is_bare_order_reference,
    is_bare_reference_attempt,
    looks_like_a_reference,
    route_intent,
)
from chatbot.rag_answer import answer_or_refuse
from config import config
from persistence import repository
from tools.order_data_tools import describe_order_for_person, fetch_order
from tools.prediction_tools import ToolError, detect_issue_mode


def new_session_id():
    """A fresh conversation identifier."""
    return "CHAT-" + uuid.uuid4().hex[:12].upper()


def _order_status_text(summary):
    return ("Order {order_id} ({items} item{plural} of {category}, INR {value:,.2f}) is with "
            "{courier} and is {position}. The last courier scan was {hours} hours ago."
            ).format(order_id=summary["order_id"], items=summary["item_count"],
                     plural="" if summary["item_count"] == 1 else "s",
                     category=summary["product_category"], value=summary["order_value_inr"],
                     courier=summary["courier"], position=summary["delivery_position"],
                     hours=summary["hours_since_last_scan"])


class CareAssistant(object):
    """Everything the chatbot needs, loaded once."""

    def __init__(self, client, knowledge, orders, issue_model, vectorizer, file_tickets=True):
        self.client = client
        self.knowledge = knowledge
        self.orders = orders
        self.issue_model = issue_model
        self.vectorizer = vectorizer
        self.file_tickets = file_tickets

    # ------------------------------------------------------------ the one entry point
    def handle_message(self, session, session_id, message_text, user_name="", user_role=""):
        memory.start_session(session, session_id, user_name)
        remembered = memory.recall(session, session_id)
        memory.record_turn(session, session_id, "customer" if user_role ==
                           config.ROLE_CUSTOMER else "user", message_text)

        result = {"answer": "", "intent": "", "intent_basis": "", "intent_reason": "",
                  "order_id": "", "order_basis": "", "citations": [], "source_documents": [],
                  "ticket_id": "", "issue_mode": "", "issue_confidence": 0.0,
                  "remembered": remembered, "refused_reason": "", "call_records": []}

        explicit = extract_order_reference(message_text)
        mistyped = looks_like_a_reference(message_text)
        if mistyped and is_bare_reference_attempt(message_text, mistyped):
            result.update({
                "answer": config.UNRECOGNISED_REFERENCE_REPLY.format(
                    mistyped, config.ORDER_REFERENCE_FORMAT),
                "intent": config.INTENT_ORDER_STATUS, "intent_basis": "reference",
                "intent_reason": "a token shaped like a reference that is not one",
                "refused_reason": "unrecognised reference " + mistyped})
            return self._finish(session, session_id, result, {})

        routed = route_intent(message_text, remembered, self.client,
                              (self.issue_model, self.vectorizer))
        if routed.get("call_record"):
            result["call_records"].append(routed["call_record"])
        result.update({"intent": routed["intent"], "intent_basis": routed["basis"],
                       "intent_reason": routed["reason"]})
        order_id, order_basis = memory.resolve_order_reference(explicit, remembered)
        result.update({"order_id": order_id, "order_basis": order_basis})
        updates = {}

        if routed["intent"] == config.INTENT_ORDER_STATUS:
            if not order_id and mistyped:
                result["answer"] = config.UNRECOGNISED_REFERENCE_REPLY.format(
                    mistyped, config.ORDER_REFERENCE_FORMAT)
                result["refused_reason"] = "unrecognised reference " + mistyped
            else:
                answer, refusal, summary = self._handle_order_status(order_id)
                result.update({"answer": answer, "refused_reason": refusal})
                if summary:
                    updates.update({"order_id": summary["order_id"],
                                    "customer_tier": summary["customer_tier"]})
        elif routed["intent"] == config.INTENT_COMPLAINT:
            filed = self._handle_complaint(session, session_id, message_text, order_id,
                                           user_name, user_role, remembered)
            result.update({"answer": filed["answer"], "ticket_id": filed["ticket_id"],
                           "issue_mode": filed["issue_mode"],
                           "issue_confidence": filed["confidence"],
                           "refused_reason": filed.get("refused_reason", "")})
            updates.update({"ticket_id": filed["ticket_id"], "issue_mode": filed["issue_mode"],
                            "customer_tier": filed["customer_tier"],
                            "order_id": filed.get("order_id", "")})
        elif routed["intent"] == config.INTENT_POLICY_QUESTION:
            question = message_text
            if remembered.get("issue_mode") and not explicit:
                question = message_text + " (about " + remembered["issue_mode"] + ")"
            # At most one call per turn: if routing spent it, answer without a call.
            client = None if result["call_records"] else self.client
            history = memory.format_history(memory.conversation_so_far(session, session_id))
            answered = answer_or_refuse(client, self.knowledge, question, history,
                                        memory.format_entities(remembered))
            if answered.get("call_record"):
                result["call_records"].append(answered["call_record"])
            result.update({"answer": answered["answer"], "citations": answered["citations"],
                           "source_documents": answered["source_documents"],
                           "refused_reason": answered["refused_reason"]})
        elif routed["intent"] == config.INTENT_SMALL_TALK:
            result["answer"] = config.SMALL_TALK_REPLY
        else:
            result["answer"] = config.OUT_OF_SCOPE_REFUSAL
            result["refused_reason"] = "out of scope: " + routed["reason"]
        return self._finish(session, session_id, result, updates)

    def _finish(self, session, session_id, result, updates):
        result["remembered"] = memory.remember(session, session_id, updates)
        memory.record_turn(session, session_id, "assistant", result["answer"],
                           intent=result["intent"], citations=result["citations"],
                           refused_reason=result["refused_reason"])
        return result

    # ------------------------------------------------------------ destinations
    def _handle_order_status(self, order_id):
        """Answer from the order record, never from a model."""
        if not order_id:
            return config.ASK_FOR_ORDER_REPLY.format(config.ORDER_REFERENCE_FORMAT), \
                "no order named or remembered", {}
        order = fetch_order(self.orders, order_id)
        if not order:
            return config.ORDER_NOT_ON_FILE_REPLY.format(order_id, config.ORDER_REFERENCE_FORMAT), \
                "order " + order_id + " is not on file", {}
        summary = describe_order_for_person(order)
        return _order_status_text(summary), "", summary

    def _handle_complaint(self, session, session_id, message_text, order_id, user_name,
                          user_role="", remembered=None):
        """File a complaint, or add to the one this conversation already filed."""
        remembered = remembered or {}
        order = fetch_order(self.orders, order_id) if order_id else {}
        if order_id and not order:
            return {"answer": config.ORDER_NOT_ON_FILE_REPLY.format(
                        order_id, config.ORDER_REFERENCE_FORMAT),
                    "issue_mode": "", "confidence": 0.0, "ticket_id": "", "customer_tier": "",
                    "refused_reason": "order " + order_id + " is not on file"}
        tier = order.get("customer_tier", "") if order else ""
        open_ticket = {}
        if remembered.get("ticket_id"):
            open_ticket = repository.find_open_ticket(session, remembered["ticket_id"])

        if is_bare_order_reference(message_text) and open_ticket:
            amended = repository.amend_ticket(session, open_ticket["ticket_id"], {
                "order_id": order.get("order_id", ""),
                "customer_id": order.get("customer_id", "")})
            return {"answer": config.TICKET_AMENDED_REPLY.format(
                        amended["ticket_id"], amended["issue_mode"], ""),
                    "issue_mode": amended["issue_mode"],
                    "confidence": amended["issue_confidence"], "ticket_id": amended["ticket_id"],
                    "customer_tier": tier, "order_id": order.get("order_id", "")}

        try:
            found = detect_issue_mode(self.issue_model, self.vectorizer, message_text)
        except ToolError:
            found = {"issue_mode": "", "confidence": 0.0}
        if found["confidence"] < config.MINIMUM_ISSUE_CONFIDENCE:
            return {"answer": config.ISSUE_UNCLEAR_REPLY, "issue_mode": "",
                    "confidence": found["confidence"], "ticket_id": "", "customer_tier": tier,
                    "refused_reason": "issue confidence {0} is below {1}".format(
                        found["confidence"], config.MINIMUM_ISSUE_CONFIDENCE)}
        suffix = "" if order else config.TICKET_NEEDS_ORDER_SUFFIX.format(
            config.ORDER_REFERENCE_FORMAT)

        if open_ticket:
            changes = {"message_text": (open_ticket["message_text"] + "\n" + message_text)[
                -config.MAXIMUM_MESSAGE_CHARACTERS:]}
            if order and not open_ticket["order_id"]:
                changes.update({"order_id": order["order_id"],
                                "customer_id": order.get("customer_id", "")})
            if found["confidence"] > float(open_ticket["issue_confidence"] or 0.0):
                changes.update({"issue_mode": found["issue_mode"],
                                "issue_confidence": found["confidence"]})
            amended = repository.amend_ticket(session, open_ticket["ticket_id"], changes)
            return {"answer": config.TICKET_AMENDED_REPLY.format(
                        amended["ticket_id"], amended["issue_mode"], suffix),
                    "issue_mode": amended["issue_mode"],
                    "confidence": amended["issue_confidence"], "ticket_id": amended["ticket_id"],
                    "customer_tier": tier, "order_id": order.get("order_id", "")}

        if not self.file_tickets:
            return {"answer": config.TICKET_NOT_FILED_REPLY.format(found["issue_mode"]),
                    "issue_mode": found["issue_mode"], "confidence": found["confidence"],
                    "ticket_id": "", "customer_tier": tier,
                    "order_id": order.get("order_id", "")}

        assigned_to = config.UNASSIGNED_QUEUE if user_role in ("", config.ROLE_CUSTOMER) \
            else user_name
        ticket_id = repository.next_ticket_identifier(session)
        repository.insert_ticket(session, {
            "ticket_id": ticket_id, "order_id": order.get("order_id", ""),
            "customer_id": order.get("customer_id", ""), "opened_by": user_name,
            "assigned_to": assigned_to, "source_channel": config.CHAT_SOURCE_CHANNEL,
            "message_text": message_text[:config.MAXIMUM_MESSAGE_CHARACTERS],
            "issue_mode": found["issue_mode"], "issue_confidence": found["confidence"],
            "status": config.TICKET_STATUS_OPEN})
        return {"answer": config.TICKET_OPENED_REPLY.format(ticket_id, found["issue_mode"],
                                                            suffix),
                "issue_mode": found["issue_mode"], "confidence": found["confidence"],
                "ticket_id": ticket_id, "customer_tier": tier,
                "order_id": order.get("order_id", "")}
