"""Document Store for Personalities."""
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.vectorstores import Chroma
from langchain.llms import LlamaCpp

class PersonalityStore:
    def __init__(self, llm: LlamaCpp) -> None:
        """Use our embedding model to create a vector store."""
        self.vectorstore = Chroma()
        self.character_names = set()

    def add_character(
        self,
        url: str,
        character_name: str,
    ) -> None:
        """Add documents to the vector store from a single URL.
        
        Args:
            url: The URL of the document to add to the vector store.
            character_name: The name of the character to add to the store.
            
        """
        if character_name in self.characters:
            return

        loader = WebBaseLoader(url)
        data = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
        all_splits = text_splitter.split_documents(data)

        self.vectorstore.add_documents(documents=all_splits)
        self.character_names.add(character_name)