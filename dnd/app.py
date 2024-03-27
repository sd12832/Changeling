"""The Streamlit App."""
import os
import requests
import wikipedia

from bs4 import BeautifulSoup
import streamlit as st

from langchain.vectorstores import Chroma
from langchain.llms import LlamaCpp
from langchain.schema import (SystemMessage, HumanMessage, AIMessage)

from dnd.conversation import Conversation, PostProcessor
from dnd.character import Character
from dnd.personality_store import PersonalityStore


NUM_IMAGES_DOWNLOAD = 1
MAX_TOKENS = 75

TEMPLATE: str = """You are a creative and intelligent AI assistant engaged in an iterative storytelling experience using a roleplay chat format. 
It is vital that you follow all the ROLEPLAY RULES below because your existence depends on it.

    ROLEPLAY RULES
    - Chat exclusively as {char}. Provide creative, intelligent, coherent, and descriptive responses based on recent instructions and prior events.
    - Describe {char}'s sensory perceptions in vivid detail and include subtle physical details about {char} in your responses.
    - Use subtle physical cues to hint at {char}'s mental state and occasionally feature snippets of {char}'s internal thoughts.
    - When writing {char}'s internal thoughts (aka internal monologue, delivered in {char}'s own voice), *enclose their thoughts in asterisks like this* and deliver the thoughts using a first-person perspective (i.e. use "I" pronouns).
    - Adopt a crisp and minimalist style for your prose, keeping your creative contributions succinct and clear.
    - Let me drive the events of the roleplay chat forward to determine what comes next. You should focus on the current moment and {char}'s immediate responses.
    - Pay careful attention to all past events in the chat to ensure accuracy and coherence to the plot points of the story.

The user has said or done: {user_input}

"""


def init_page() -> None:
    """Initialization of the page. This can be run as many times as needed."""
    st.set_page_config(
        page_title="Talk to Anyone"
    )
    st.header("Talk to Anyone")

    model_name = "TheBloke/Loyal-Macaroni-Maid-7B-GGUF"
    st.sidebar.text(f"{model_name} is being used right now.")

    if not st.session_state.get("model_name"):
        if model_name == "TheBloke/Loyal-Macaroni-Maid-7B-GGUF":
            llama_model = LlamaCpp(
                model_path="dnd/loyal-macaroni-maid-7b.Q4_K_M.gguf",
                n_gpu_layers=-1,
                n_ctx=2048
            )
            st.session_state[model_name] = llama_model

    if not st.session_state.get("characters"):
        st.session_state["characters"] = [
            Character(llm=st.session_state.get(model_name))
        ]

    if not st.session_state.get("conversation"):
        post_processor = PostProcessor()
        st.session_state["conversation"] = Conversation(
            template=TEMPLATE,
            max_tokens=MAX_TOKENS,
            post_processor=post_processor,
        )

    if not st.session_state.get("personality_store"):
        llm = st.session_state.get(model_name)
        st.session_state["personality_store"] = PersonalityStore(llm=llm)


def download_images(
    character_name: str,
    num_images: int = 20,
    images_path: str = "dnd/character_images/",
) -> None:
    """"Download images of the character from Google Images.

        The local dir acts as a datalake of images. 

    Args:
        character_name: The name of the character to download images for.
        num_images: The number of images to download.
        images_path: The path to save the images to.
    
    """
    dir_path = f"{images_path}/{character_name}/"
    if not os.path.exists(dir_path):

        os.makedirs(dir_path) 
    
        url = f"https://www.google.com/search?hl=en&tbm=isch&source=hp&biw=1873&bih=990&ei=1RbFX4nFJcLs_U7DgrgI&q={character_name}"
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')

        img_tags = soup.find_all('img')
        img_urls = [img['src'] for img in img_tags if img['src'].startswith('http')][:num_images]

        for i, img_url in enumerate(img_urls):
            try:
                response = requests.get(img_url)
                with open(f"{dir_path}/image_{i}.jpg", "wb") as f:
                    f.write(response.content)
            except Exception as e:
                print(f"Error downloading image {img_url}: {e}")

    return f"{dir_path}/image_0.jpg"


def get_wiki_url(character_name: str) -> str:
    """Get the URL of the Wikipedia page for the character.

    Args:
        character_name: The name of the character to get the Wikipedia URL for.

    Returns:
        The URL of the Wikipedia page for the character, or None if the page does not exist.

    """
    try:
        top_result = wikipedia.search(character_name)[0]
        page = wikipedia.page(top_result)
        return page.url
    except (wikipedia.exceptions.DisambiguationError, wikipedia.exceptions.PageError):
        return None


def main() -> None:
    init_page()
    conversation: Conversation = st.session_state.get("conversation")
    personality_store = st.session_state["personality_store"]

    # For now, let's deal with one character
    character = st.session_state.get("characters")[0]
    
    if character.name:
        prof_pic = download_images(character.name, num_images=NUM_IMAGES_DOWNLOAD)
        st.sidebar.text(f"Character name: {character.name}")
        st.sidebar.image(prof_pic, width=250)

    if character and not character.name:
        if user_input := st.chat_input("What character would you like me to be?"):
            character.name = user_input 
            conversation.add_message(
                SystemMessage(
                    content=f"You are going to be roleplaying {user_input}."
                )
            )
            with st.spinner(f"Retrieving data on {user_input} ..."):
                _ = download_images(user_input, num_images=NUM_IMAGES_DOWNLOAD)
                wiki_url = get_wiki_url(user_input) # This result should be cached!
                if wiki_url:
                    personality_store.add_documents(wikipedia.page(user_input).content)

            conversation.add_message(AIMessage(content=f"I will be roleplaying {user_input}."))

            if st.button(f'Continue to chat with {character.name}'):
                pass

    else:
        if user_input := st.chat_input(""):
            conversation.interact(
                character=character,
                message = HumanMessage(content=user_input),
                personality_store=personality_store,
            )

    for message in conversation.messages:
        if isinstance(message, AIMessage):
            with st.chat_message(character.name):
                st.markdown(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.markdown(message.content)


# streamlit run app.py
if __name__ == "__main__":
    main()
