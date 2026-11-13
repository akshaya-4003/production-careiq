"""Entry point 6: start the REST service on config.API_PORT.

    python3 run_api_service.py

Entry points 2 to 5 must have run first: this loads the three models and the
policy index when it starts.
"""

import os
import sys

import uvicorn

from config import config

REQUIRED = [
    (config.APPLICATION_DATABASE_FILE, "python3 run_database_setup.py"),
    (config.CLEAN_LIVE_ORDERS_FILE, "python3 run_data_preparation.py"),
    (config.ESCALATION_MODEL_FILE, "python3 run_model_training.py"),
    (config.VECTOR_STORE_DIRECTORY, "python3 run_knowledge_indexing.py"),
]


def main():
    for path, command in REQUIRED:
        if not os.path.exists(path) or (os.path.isfile(path) and os.path.getsize(path) == 0):
            print("Missing " + path + ". Run " + command + " first.")
            return 1

    from api.routes import create_api_application

    application = create_api_application(config.API_TITLE)
    context = application.state.context
    print("Starting CareIQ API")
    print("    orders monitored   : " + str(len(context.orders)))
    print("    policy chunks      : " + str(context.knowledge.count()))
    print("    language model     : " + context.client.name
          + ((" (" + context.client.model_name + ")") if context.client.model_name else ""))
    print("    embeddings         : " + context.knowledge.backend_name())
    print("    docs at http://localhost:" + str(config.API_PORT) + "/docs")
    for account in config.DEMO_USERS:
        print("    sign in: {0:<8} / {1:<14} ({2})".format(account["user_name"],
                                                         account["password"], account["role"]))
    uvicorn.run(application, host=config.API_HOST, port=config.API_PORT, log_level="warning")
    return 0


if __name__ == "__main__":
    sys.exit(main())
