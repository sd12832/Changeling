import re

from dataclasses import dataclass, field
from langchain.schema import (SystemMessage, HumanMessage, AIMessage)
from typing import Any
from langchain import PromptTemplate

from dnd.character import Character
from dnd.personality_store import PersonalityStore


class PostProcessor:
    """Class to post process the output of a character chat message."""
    # TODO: Use a Protocol here instead of the actual class
    def __init__(self) -> None:
        pass

    def simple_process(self, message: str) -> str:
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', message)
        if sentences[-1][-1] not in ".!?":
            sentences.pop()

        return ' '.join(sentences)


@dataclass
class Conversation:
    """Conversation object to store the messages of the chat."""
    messages: list[SystemMessage | HumanMessage | AIMessage] = field(default_factory=list)
    llama_messages: list[dict[str, Any]] = field(default_factory=list)

    def __init__(
        self,
        template: str,
        max_tokens: int = 75,
        post_processor: PostProcessor = PostProcessor(),
    ) -> None:
        """Initialize the chat object"""
        self.messages = []
        self.llama_messages = []
        self.ai_prompt = PromptTemplate(
            input_variables=["char", "user_input"],
            template=template
        )
        self.max_tokens = max_tokens
        self.post_processor = post_processor

    def __str__(self) -> str:
        """Return a string representation of the Conversation object."""
        messages = "\n".join([f"{i+1}. {message}" for i, message in enumerate(self.messages)])
        return f"Messages:\n{messages}"
    
    def add_message(self, message: SystemMessage | HumanMessage | AIMessage) -> None:
        """Add a message to the chat object."""
        self.messages.append(message)
        self.llama_messages.append(self.convert_langchainschema_to_dict(message))

    def interact(
        self,
        character: Character,
        message: HumanMessage,
        personality_store: PersonalityStore,
    ) -> None:
        """Add a human message to the chat object. Also determines the answer.

            Currently only supports intereacting with a single character/agent.
            A dynamic interaction with multiple characters is the challenge here.

        Args:
            character: The character object that is being roleplayed.
            message: The message object that the user has sent.
            personality_store: The personality store object that contains the character's data.
        
        """
        prompt_formatted_message = self.ai_prompt.format(
            char=character.name,
            user_input=message.content
        )
        prompted_message = HumanMessage(content=prompt_formatted_message)

        self.messages.append(message)
        self.llama_messages.append(self.convert_langchainschema_to_dict(prompted_message))

        # answer = character.llm.create_chat_completion(
        #     messages=self.llama_messages,
        #     max_tokens=self.max_tokens,
        # )

        

        # Clean up the answer
        answer_content = answer["choices"][0]["message"]["content"]
        sentences = self.post_processor.simple_process(answer_content)

        self.llama_messages.pop()
        self.llama_messages.append(self.convert_langchainschema_to_dict(message))

        self.add_message(AIMessage(content=sentences))

    @staticmethod
    def find_role(message: SystemMessage | HumanMessage | AIMessage) -> str:
        """Identify role name from langchain.schema object."""
        if isinstance(message, SystemMessage):
            return "system"
        if isinstance(message, HumanMessage):
            return "user"
        if isinstance(message, AIMessage):
            return "assistant"
        raise TypeError("Unknown message type.")
    
    @staticmethod
    def convert_langchainschema_to_dict(
        message: SystemMessage | HumanMessage | AIMessage,
    ) -> list[dict[str, Any]]:
        """Convert a single langchain.schema to dictionaries for ingestion by the model."""
        return {
            "role": Conversation.find_role(message),
            "content": message.content
        }