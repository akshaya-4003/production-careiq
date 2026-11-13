"""Six agents, each one language model call over facts it did not compute.

Every prompt states its facts as indented 'Label: value' lines and asks for a
labelled reply. No agent computes a probability, a priority, a cost, or whether a
human is needed: each of those arrives as a fact.
"""

from config import config
from llm.client import ask_for_labels

_RULES = (
    "Rules:\n"
    "- Use only the facts below. Do not compute, estimate or change any figure.\n"
    "- Reply with exactly the labelled lines requested, each as LABEL: value.\n")


def _facts_block(facts):
    lines = ["Facts:"]
    for label, value in facts.items():
        text = str(value).replace("\n", " ").strip()
        lines.append("  {0}: {1}".format(label, text))
    return "\n".join(lines)


def _order_facts(order_summary):
    return {
        "Order": order_summary.get("order_id", ""),
        "Product category": order_summary.get("product_category", ""),
        "Item count": order_summary.get("item_count", ""),
        "Order value (INR)": order_summary.get("order_value_inr", ""),
        "Courier": order_summary.get("courier", ""),
        "Customer tier": order_summary.get("customer_tier", ""),
        "Payment method": order_summary.get("payment_method", ""),
        "Delivery position": order_summary.get("delivery_position", ""),
        "Hours since last scan": order_summary.get("hours_since_last_scan", ""),
    }


def _prompt(role_line, facts, task, labels, sections=None, context_text=""):
    shape = "\n".join(label + ": <...>" for label in labels)
    if sections:
        shape += "\n" + "\n".join(section + ":\n1. <step>\n2. <step>" for section in sections)
    parts = [role_line, _RULES, _facts_block(facts)]
    if context_text:
        parts.append(context_text)
    parts.extend(["Task: " + task, "Reply format:\n" + shape])
    return "\n\n".join(parts)


def triage_planner_agent(client, order_summary, request_text):
    """Decide how to look into this order. The risk analyst reads the plan."""
    labels = ["PLAN", "FOCUS"]
    facts = _order_facts(order_summary)
    facts["Request"] = request_text
    prompt = _prompt("You are the triage planner on the Northlane care desk.", facts,
                     "State in one or two sentences how this order should be looked into, "
                     "and what the analysts should focus on.", labels)
    return ask_for_labels(client, prompt, config.AGENT_NAME_TRIAGE, labels)


def risk_analyst_agent(client, order_summary, risk_facts, triage_plan):
    """Explain the score the model already produced. It must not recalculate it."""
    labels = ["RISK SUMMARY", "KEY DRIVERS"]
    facts = _order_facts(order_summary)
    facts.update({
        "Escalation probability": risk_facts.get("escalation_probability", ""),
        "Risk band": risk_facts.get("risk_band", ""),
        "Priority": risk_facts.get("priority", ""),
        "Severity code": risk_facts.get("severity_code", ""),
        "Triage plan": triage_plan.get("PLAN", ""),
    })
    prompt = _prompt("You are the risk analyst. The escalation model has already scored this "
                     "order. You must not recalculate, adjust or second-guess the probability, "
                     "the band or the priority; explain them.", facts,
                     "Explain in plain words why this order carries this risk.", labels)
    return ask_for_labels(client, prompt, config.AGENT_NAME_RISK, labels)


def issue_analyst_agent(client, order_summary, issue_facts):
    """Judge the classifier's reading of the message; it may disagree."""
    labels = ["AGREES", "ISSUE MODE", "REASONING"]
    facts = _order_facts(order_summary)
    facts.update({
        "Customer message": issue_facts.get("message_text", ""),
        "Classifier issue mode": issue_facts.get("issue_mode", ""),
        "Classifier confidence": issue_facts.get("confidence", ""),
        "Allowed issue modes": ", ".join(config.VALID_ISSUE_MODES),
    })
    prompt = _prompt("You are the issue analyst.", facts,
                     "Say YES or NO under AGREES for whether the message describes the "
                     "classifier's issue mode, name the issue mode you read (one of the "
                     "allowed modes), and give one sentence of reasoning.", labels)
    return ask_for_labels(client, prompt, config.AGENT_NAME_ISSUE, labels)


def policy_advisor_agent(client, issue_mode, order_summary, context_text):
    """Read the retrieved passages and name one remedy and its policy basis, with citations."""
    labels = ["REMEDY", "POLICY BASIS", "CITATIONS"]
    facts = _order_facts(order_summary)
    facts.update({
        "Issue mode": issue_mode,
        "Allowed remedies": ", ".join(config.VALID_REMEDY_TYPES),
        "Usual remedy for this issue": config.DEFAULT_REMEDY_BY_ISSUE.get(issue_mode, ""),
    })
    prompt = _prompt("You are the policy advisor. You may rely only on the retrieved passages "
                     "between the markers below.", facts,
                     "Name exactly one remedy from the allowed list, state the policy basis "
                     "in one sentence, and cite the passages you relied on by their labels in "
                     "square brackets, for example [CH0001].", labels, context_text=context_text)
    return ask_for_labels(client, prompt, config.AGENT_NAME_POLICY, labels)


def resolution_composer_agent(client, order_summary, risk_assessment, issue_analysis,
                              policy_guidance, costed):
    """Write the customer reply and the action steps from the facts it is handed."""
    labels = ["CUSTOMER REPLY"]
    sections = ["ACTION STEPS"]
    facts = _order_facts(order_summary)
    facts.update({
        "Issue mode": costed.get("issue_mode", ""),
        "Remedy": costed.get("remedy_type", ""),
        "Cash committed (INR)": costed.get("remedy_amount_inr", ""),
        "Estimated resolution hours": costed.get("estimated_resolution_hours", ""),
        "Needs manager approval": costed.get("needs_approval", ""),
        "Policy basis": policy_guidance.get("POLICY BASIS", ""),
        "Risk summary": risk_assessment.get("RISK SUMMARY", ""),
        "Issue reasoning": issue_analysis.get("REASONING", ""),
    })
    prompt = _prompt("You are the resolution composer. Tone follows NRC-CSC-014: plain "
                     "language, short sentences, state what will happen next. Never promise "
                     "anything beyond the remedy and figures given.", facts,
                     "Write the reply to the customer and the numbered action steps for the "
                     "desk.", labels, sections)
    return ask_for_labels(client, prompt, config.AGENT_NAME_COMPOSER, labels, sections)


def resolution_finalizer_agent(client, draft, decision):
    """Rewrite the reply to carry out the manager's instruction and say what changed."""
    labels = ["FINAL REPLY", "WHAT CHANGED"]
    sections = ["ACTION STEPS"]
    facts = {
        "Order": draft.get("order_id", ""),
        "Issue mode": draft.get("issue_mode", ""),
        "Policy basis": draft.get("policy_basis", ""),
        "Decision": decision.get("decision", ""),
        "Manager feedback": decision.get("feedback_text", ""),
        "Draft remedy": draft.get("remedy_type", ""),
        "Final remedy": decision.get("final_remedy_type", draft.get("remedy_type", "")),
        "Final cash committed (INR)": decision.get("final_amount_inr",
                                                   draft.get("remedy_amount_inr", "")),
        "Final estimated hours": decision.get("final_estimated_hours",
                                              draft.get("estimated_resolution_hours", "")),
        "Draft reply": draft.get("customer_reply", ""),
        "Draft action steps": draft.get("action_steps", "").replace("\n", " | "),
    }
    prompt = _prompt("You are the resolution finalizer.", facts,
                     "Rewrite the draft reply and the numbered action steps so they carry out "
                     "the manager's decision and feedback with the final remedy, then state "
                     "briefly what you changed.", labels, sections)
    return ask_for_labels(client, prompt, config.AGENT_NAME_FINALIZER, labels, sections)
