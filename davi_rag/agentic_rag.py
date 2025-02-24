from davi_rag.tools import SearchTool
from davi_rag.vectordb import ChromaDB
from langchain_openai import ChatOpenAI
from davi_rag.embeddings import Embedder
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent, AgentType


class AgenticRag:

    def __init__(self,
                 chroma_db: ChromaDB,
                 embedder: Embedder
                 ):
        search_tool = SearchTool(chroma_db=chroma_db,
                                 embedder=embedder)
        llm = ChatOpenAI(temperature=0)
        memory = ConversationBufferMemory(memory_key="chat_history",
                                          return_messages=True)
        self.agent = initialize_agent(
            tools=[search_tool],
            llm=llm,
            agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
            memory=memory,
            verbose=True,
            handle_parsing_errors=True
        )

    def run(self, query: str):

        return self.agent.run(query)
