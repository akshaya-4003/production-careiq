"""Entry point 5: chunk and embed the policy book, then prove it answers real questions.

    python3 run_knowledge_indexing.py
"""

import argparse
import sys

from config import config
from knowledge.embedding import load_embedder
from knowledge.procedure_retrieval import open_policy_knowledge, topics_in_question
from knowledge.vector_index import build_policy_index, open_policy_index

PROOF_QUESTIONS = [
    "how many days do I have to return a pair of shoes",
    "what compensation is due when a delivery is late",
    "who may approve a goodwill credit",
    "my parcel has not moved for days",
]


def main():
    parser = argparse.ArgumentParser(description="Build the policy index.")
    parser.add_argument("--backend", default=None,
                        help="offline, minilm or auto (default: config.EMBEDDING_BACKEND)")
    arguments = parser.parse_args()

    print("Building the Northlane policy index")
    try:
        count, embedder = build_policy_index(embedder=load_embedder(arguments.backend))
    except FileNotFoundError as error:
        print("    " + str(error))
        return 1
    collection = open_policy_index()
    print("    embedding backend      : " + str(embedder.describe()))
    print("    chunks stored          : " + str(count))
    print("    store location         : " + config.VECTOR_STORE_DIRECTORY)
    print("    collection metadata    : " + str(collection.metadata))

    print("Proving the index answers real questions")
    knowledge = open_policy_knowledge(embedder=embedder)
    for question in PROOF_QUESTIONS:
        print("  Q: " + question)
        print("     topics matched: " + str(topics_in_question(question)))
        for chunk in knowledge.search(question, use_topic_filter=False):
            print("     {0:.4f}  [{1}] {2} / {3}".format(chunk["distance"], chunk["chunk_id"],
                                                         chunk["document_id"], chunk["heading"]))
    print("Done. Next: python3 run_api_service.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
