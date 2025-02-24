import os
import hashlib
import chromadb
from typing import List
from langchain_core.documents.base import Document


class ChromaDB:

    def __init__(self,
                 persist_directory: str,
                 collection_name: str):
        os.makedirs(persist_directory, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=collection_name
        )

    @staticmethod
    def _generate_sha_id(text):
        """
        Generates a unique SHA256 hash ID for a given text.
        """
        return hashlib.sha256(text.encode()).hexdigest()

    def add_documents(self,
                      documents: List[Document]):

        for document in documents:
            doc_text = document.page_content
            doc_id = self._generate_sha_id(doc_text)
            embeddings = document.metadata.get("embeddings")
            if not embeddings:
                print("[WARNING] skipping document NO EMBEDDINGS FOUND")
            self.collection.add(
                ids=[doc_id],
                embeddings=[embeddings],
                metadatas=[{"text": doc_text}]
            )

    def search_by_embedding(self,
                            embedding: List[float],
                            top_k: int = 5) -> List[dict]:
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=top_k
        )

        matches = []
        for i in range(len(results['ids'][0])):
            match = {
                "id": results['ids'][0][i],
                "metadata": results['metadatas'][0][i],
                "distance": results['distances'][0][i]
            }
            matches.append(match)

        return matches
