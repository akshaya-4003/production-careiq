"""Entry point 4: train thirteen candidates and choose three.

    python3 run_model_training.py
"""

import sys
import warnings

from config import config
from data_pipeline.dataset_loader import load_processed_dataset_if_present
from modeling.escalation_risk_model import escalation_candidates, train_escalation_model
from modeling.issue_mode_model import train_issue_mode_model
from modeling.model_evaluation import save_evaluation_report
from modeling.resolution_time_model import train_resolution_model

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def main():
    orders = load_processed_dataset_if_present(config.CLEAN_ORDERS_TRAIN_FILE)
    tickets = load_processed_dataset_if_present(config.CLEAN_TICKETS_TRAIN_FILE)
    messages = load_processed_dataset_if_present(config.CLEAN_MESSAGES_FILE)
    if orders is None or tickets is None or messages is None:
        print("The cleaned tables are missing. Run python3 run_data_preparation.py first.")
        return 1

    escalation_count = len(escalation_candidates())
    print("Model 1 of 3 - escalation risk classifier ({0} candidates)".format(escalation_count))
    escalation = train_escalation_model(orders)
    print("Model 2 of 3 - resolution time regressor (3 candidates)")
    resolution = train_resolution_model(tickets)
    print("Model 3 of 3 - issue mode classifier (3 candidates, text)")
    issue = train_issue_mode_model(messages)

    def summary(result):
        return {key: value for key, value in result.items()
                if key not in ("model", "vectorizer")}

    metrics = {
        "chosen_models": {
            "escalation_risk": escalation["winner"],
            "resolution_time": resolution["winner"],
            "issue_mode": issue["winner"],
        },
        "escalation_risk": summary(escalation),
        "resolution_time": summary(resolution),
        "issue_mode": summary(issue),
    }
    written = save_evaluation_report(metrics, config.MODEL_METRICS_FILE)
    print("Metrics written: " + written)
    print("Chosen models:")
    print("    escalation risk : " + escalation["winner"])
    print("    resolution time : " + resolution["winner"])
    print("    issue mode      : " + issue["winner"])
    print("Done. Next: python3 run_knowledge_indexing.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
