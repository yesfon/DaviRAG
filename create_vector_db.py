from davi_rag.vectordb import ChromaDB
from davi_rag.embeddings import Embedder
from langchain.text_splitter import TokenTextSplitter
from langchain_community.document_loaders import PyPDFLoader


# Load and Chunk file
chunk_size = 1024
chunk_overlap = 50
file_path = ("data/1210-Insurance-2030-"
             "The-impact-of-AI-on-the-future-of-insurance"
             "-_-McKinsey-Company.pdf")
loader = PyPDFLoader(file_path)
documents = loader.load()
text_splitter = TokenTextSplitter(
    chunk_size=chunk_size, chunk_overlap=chunk_overlap
)
chunks = text_splitter.split_documents(documents)

# Get embeddings
embedder = Embedder(model_name="text-embedding-3-large")
print("Getting embeddings")
for chunk in chunks:
    chunk.metadata["embeddings"] = embedder.get_embeddings(chunk.page_content)

# Add to vector DB
vector_db = ChromaDB(persist_directory="./chroma_db",
                     collection_name="davivienda")
vector_db.add_documents(documents=chunks)
