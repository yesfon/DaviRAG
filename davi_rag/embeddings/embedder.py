from typing import List
from langchain_openai import OpenAIEmbeddings


class Embedder:

    def __init__(self, model_name: str = "text-embedding-3-large"):
        self._model_name = model_name
        self.embedding_model = OpenAIEmbeddings(model=self._model_name)

    def get_embeddings(self, text: str) -> List[float]:
        return self.embedding_model.embed_query(text)
