from davi_rag.vectordb import ChromaDB
from davi_rag.embeddings import Embedder
from davi_rag.agentic_rag import AgenticRag
from frontapp.frontapp import FrontApp


persist_directory = "./chroma_db"
collection_name = "davivienda"

chroma_db = ChromaDB(persist_directory=persist_directory,
                     collection_name=collection_name)
embedder = Embedder()

agent = AgenticRag(
    chroma_db=chroma_db,
    embedder=embedder
)

front_app = FrontApp(agent)
front_app.run()
