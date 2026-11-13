"""Writes chunks into the on-disk store and searches them."""

import os
import shutil

import chromadb

from config import config
from knowledge.chunk_splitting import chunk_documents
from knowledge.embedding import load_embedder
from knowledge.pdf_extraction import load_policy_documents

_METADATA_FIELDS = ("chunk_id", "document_id", "document_title", "topic", "heading")


def open_client(store_path):
    """Open the on-disk store, creating its parent folder when absent."""
    parent = os.path.dirname(os.path.abspath(store_path))
    os.makedirs(parent, exist_ok=True)
    return chromadb.PersistentClient(path=store_path)


def build_policy_index(documents_directory=None, store_path=None, collection_name=None,
                       embedder=None):
    """Read, chunk, embed and write the store from scratch, pinned to cosine distance."""
    folder = config.DOCS_DIRECTORY if documents_directory is None else documents_directory
    store = config.VECTOR_STORE_DIRECTORY if store_path is None else store_path
    name = config.VECTOR_COLLECTION_NAME if collection_name is None else collection_name
    model = load_embedder() if embedder is None else embedder

    chunks = chunk_documents(load_policy_documents(folder))
    if not chunks:
        raise ValueError("No chunks were produced from the documents in " + folder)

    client = open_client(store)
    # Delete first: a get-or-create would hand back the old metric and old vectors.
    existing = [getattr(item, "name", item) for item in client.list_collections()]
    if name in existing:
        client.delete_collection(name)
    collection = client.create_collection(
        name=name,
        metadata={"hnsw:space": "cosine", "embedding_backend": model.name},
        embedding_function=None,
    )
    texts_to_embed = [chunk["heading"] + ". " + chunk["text"] for chunk in chunks]
    vectors = model.encode(texts_to_embed)
    collection.add(
        ids=[chunk["chunk_id"] for chunk in chunks],
        documents=[chunk["text"] for chunk in chunks],
        embeddings=[list(map(float, vector)) for vector in vectors],
        metadatas=[{field: chunk[field] for field in _METADATA_FIELDS} for chunk in chunks],
    )
    return len(chunks), model


def open_policy_index(store_path=None, collection_name=None):
    """Open the collection a previous build left behind; a missing one raises."""
    store = config.VECTOR_STORE_DIRECTORY if store_path is None else store_path
    name = config.VECTOR_COLLECTION_NAME if collection_name is None else collection_name
    if not os.path.isdir(store):
        raise FileNotFoundError("The policy index does not exist at " + store
                                + ". Run python3 run_knowledge_indexing.py first.")
    client = open_client(store)
    try:
        return client.get_collection(name=name, embedding_function=None)
    except Exception as error:
        raise FileNotFoundError("The policy collection " + name + " does not exist in " + store
                                + ". Run python3 run_knowledge_indexing.py first. (" + str(error)
                                + ")")


def index_backend_name(collection):
    """The embedder recorded when the index was built, or 'unknown'."""
    metadata = collection.metadata or {}
    return str(metadata.get("embedding_backend", "unknown"))


def delete_policy_index(store_path=None):
    """Remove the whole store so a rebuild starts clean."""
    store = config.VECTOR_STORE_DIRECTORY if store_path is None else store_path
    if os.path.isdir(store):
        shutil.rmtree(store)
    return store


def search_policy(collection, question, embedder, how_many=None, where=None):
    """The nearest chunks to a question, each carrying everything a citation needs."""
    count = config.RETRIEVAL_RESULT_COUNT if how_many is None else how_many
    query_vector = [float(value) for value in embedder.encode_one(question)]
    arguments = {"query_embeddings": [query_vector], "n_results": int(count),
                 "include": ["documents", "metadatas", "distances"]}
    if where:
        arguments["where"] = where
    answer = collection.query(**arguments)
    results = []
    for chunk_id, text, metadata, distance in zip(answer["ids"][0], answer["documents"][0],
                                                   answer["metadatas"][0],
                                                   answer["distances"][0]):
        results.append({
            "chunk_id": chunk_id,
            "text": text,
            "heading": metadata.get("heading", ""),
            "document_id": metadata.get("document_id", ""),
            "document_title": metadata.get("document_title", ""),
            "topic": metadata.get("topic", ""),
            "distance": round(float(distance), config.EMBEDDING_DISTANCE_DECIMALS),
        })
    return results
