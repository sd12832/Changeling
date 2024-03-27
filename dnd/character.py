from dataclasses import dataclass
from langchain.llms import LlamaCpp

@dataclass
class Character:
    """Character object to store the context of the chat. Analogous to Agent."""
    llm: LlamaCpp | None = None
    name: str | None = None
    in_db: bool = False

    def __init__(self, llm: LlamaCpp) -> None:
        """Initialize the chat object"""
        self.llm = llm

    def __str__(self) -> str:
        """Return a string representation of the Character object."""
        return f"Character Name: {self.name} with model {self.llm}"
