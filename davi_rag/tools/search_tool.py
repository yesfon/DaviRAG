from pydantic import PrivateAttr
from davi_rag.vectordb import ChromaDB
from davi_rag.embeddings import Embedder
from langchain_core.tools import BaseTool


class SearchTool(BaseTool):
    name: str = "search"
    description: str = (
        "Searches the indexed documents using semantic similarity. "
        "Provide a query text and the tool will return the most "
        "relevant documents."
    )

    # Declare private attributes for non-field values.
    _chroma_db: ChromaDB = PrivateAttr()
    _embedder: Embedder = PrivateAttr()

    def __init__(self,
                 chroma_db: ChromaDB,
                 embedder: Embedder,
                 **kwargs):
        super().__init__(**kwargs)
        self._chroma_db = chroma_db
        self._embedder = embedder

    def _run(self, input: str) -> str:
        embedding = self._embedder.get_embeddings(input)
        results = self._chroma_db.search_by_embedding(embedding)
        if not results:
            return "No relevant documents found."
        result_lines = []
        for res in results:
            line = (
                f"ID: {res['id']}\n"
                f"Text: {res['metadata'].get('text')}\n"
                f"Distance: {res['distance']:.4f}\n"
            )
            result_lines.append(line)
        return "\n".join(result_lines)

    async def _arun(self,
                    input: str) -> str:
        raise NotImplementedError("SearchTool does not support async "
                                  "execution.")
