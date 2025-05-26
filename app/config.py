import os
import streamlit as st
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

# Default Model Configurations 
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_TEXT_GENERATION_MODEL = "gpt-4o" 
DEFAULT_IMAGE_GENERATION_MODEL = "dall-e-3" 

# Available Models for User Selection 
EMBEDDING_MODELS = {
    "OpenAI (text-embedding-ada-002)": "text-embedding-ada-002",
    "OpenAI (text-embedding-3-small)": "text-embedding-3-small", 
    "OpenAI (text-embedding-3-large)": "text-embedding-3-large", 
    # "Sentence Transformers (Local)": "sentence-transformers/all-MiniLM-L6-v2" # Placeholder for future open-source integration
}
TEXT_GENERATION_MODELS = {
    "GPT-4o": "gpt-4o",
    "GPT-4 Turbo": "gpt-4-turbo",
    "GPT-4": "gpt-4",
    "GPT-3.5 Turbo": "gpt-3.5-turbo",
}

IMAGE_GENERATION_MODELS = {
    "DALL-E 3": "dall-e-3",
    "DALL-E 2": "dall-e-2",
}

# Default Output Tone
DEFAULT_TONE = "Funny"
TONE_OPTIONS = ["Funny", "Narrator", "Whimsical", "Sarcastic", "Formal"]

# Paths 
DATA_DIR = "data/stories"
EMBEDDINGS_DIR = "embeddings"
FAISS_INDEX_PATH = os.path.join(EMBEDDINGS_DIR, "story_embeddings.faiss")
TEXT_CHUNKS_PATH = os.path.join(EMBEDDINGS_DIR, "story_chunks.json") 
