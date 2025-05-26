import streamlit as st
import os
# from dotenv import load_dotenv
# load_dotenv()

from app.retriever import create_and_store_embeddings, load_faiss_index_and_chunks 
from app.main import process_query
from app.config import (
    OPENAI_API_KEY,
    EMBEDDING_MODELS,
    TEXT_GENERATION_MODELS,
    IMAGE_GENERATION_MODELS,
    TONE_OPTIONS,
    DEFAULT_TONE,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_TEXT_GENERATION_MODEL,
    DEFAULT_IMAGE_GENERATION_MODEL,
    DATA_DIR,
    FAISS_INDEX_PATH, # We will use this path in session state
    TEXT_CHUNKS_PATH
)

# --- Page Configuration ---
st.set_page_config(
    page_title="Storyteller AI",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📚 Whimsical Storyteller AI")
st.markdown("Ask me anything about Alice in Wonderland, Gulliver's Travels, or The Arabian Nights, and I'll reply with a funny story and an image!")

# --- API Key Check ---
if not OPENAI_API_KEY:
    st.error("OpenAI API Key not found. For local development, ensure `OPENAI_API_KEY` is set in your `.env` file. For Streamlit Cloud, add it to `st.secrets`.")
    st.stop()

# --- Initialize Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = []
# Store the path to the FAISS index, not the object itself
if "faiss_index_path" not in st.session_state:
    st.session_state.faiss_index_path = None
if "all_chunks" not in st.session_state:
    st.session_state.all_chunks = []
if "embeddings_built" not in st.session_state:
    st.session_state.embeddings_built = False

# --- Sidebar for Configuration ---
st.sidebar.header("⚙️ Settings")

# Output Tone Control
st.sidebar.subheader("Output Tone")
selected_tone = st.sidebar.selectbox(
    "Choose the AI's storytelling tone:",
    TONE_OPTIONS,
    index=TONE_OPTIONS.index(st.session_state.get("selected_tone", DEFAULT_TONE)),
    key="tone_selector"
)
st.session_state.selected_tone = selected_tone # Store in session state

# Model Selection
st.sidebar.subheader("Model Selection")

# Embedding Model
selected_embedding_model_display = st.sidebar.selectbox(
    "Embedding Model:",
    list(EMBEDDING_MODELS.keys()),
    index=list(EMBEDDING_MODELS.keys()).index(
        next((k for k, v in EMBEDDING_MODELS.items() if v == st.session_state.get("selected_embedding_model", DEFAULT_EMBEDDING_MODEL)), list(EMBEDDING_MODELS.keys())[0])
    ),
    key="embedding_model_selector"
)
selected_embedding_model = EMBEDDING_MODELS[selected_embedding_model_display]
st.session_state.selected_embedding_model = selected_embedding_model

# Text Generation Model
selected_text_gen_model_display = st.sidebar.selectbox(
    "Text Generation Model:",
    list(TEXT_GENERATION_MODELS.keys()),
    index=list(TEXT_GENERATION_MODELS.keys()).index(
        next((k for k, v in TEXT_GENERATION_MODELS.items() if v == st.session_state.get("selected_text_gen_model", DEFAULT_TEXT_GENERATION_MODEL)), list(TEXT_GENERATION_MODELS.keys())[0])
    ),
    key="text_gen_model_selector"
)
selected_text_gen_model = TEXT_GENERATION_MODELS[selected_text_gen_model_display]
st.session_state.selected_text_gen_model = selected_text_gen_model

# Image Generation Model
selected_image_gen_model_display = st.sidebar.selectbox(
    "Image Generation Model:",
    list(IMAGE_GENERATION_MODELS.keys()),
    index=list(IMAGE_GENERATION_MODELS.keys()).index(
        next((k for k, v in IMAGE_GENERATION_MODELS.items() if v == st.session_state.get("selected_image_gen_model", DEFAULT_IMAGE_GENERATION_MODEL)), list(IMAGE_GENERATION_MODELS.keys())[0])
    ),
    key="image_gen_model_selector"
)
selected_image_gen_model = IMAGE_GENERATION_MODELS[selected_image_gen_model_display]
st.session_state.selected_image_gen_model = selected_image_gen_model

# --- Knowledge Base Management ---
st.sidebar.subheader("Knowledge Base")

# PDF Uploader
uploaded_files = st.sidebar.file_uploader(
    "Upload PDF Story Files",
    type="pdf",
    accept_multiple_files=True,
    help="Upload your story PDFs here. These will be used to build the knowledge base, prioritizing them over files in 'data/stories/'."
)

# Button to rebuild embeddings
if st.sidebar.button("Rebuild Story Knowledge Base"):
    # Clear existing index path and chunks in session state before rebuilding
    st.session_state.faiss_index_path = None
    st.session_state.all_chunks = []
    st.session_state.embeddings_built = False

    with st.spinner("Building knowledge base... This might take a moment!"):
        # create_and_store_embeddings will save the index and chunks to disk
        faiss_index_obj, all_chunks_list = create_and_store_embeddings(
            st.session_state.selected_embedding_model,
            uploaded_files=uploaded_files # Pass uploaded files
        )
        
        if faiss_index_obj is not None and all_chunks_list:
            st.session_state.faiss_index_path = FAISS_INDEX_PATH # Store the path
            st.session_state.all_chunks = all_chunks_list
            st.session_state.embeddings_built = True
            st.sidebar.success("Knowledge Base Built Successfully!")
        else:
            st.sidebar.error("Failed to build Knowledge Base. Check console for errors. Ensure PDFs are valid.")

# Load existing embeddings on app start if not already loaded
# We only check if the files exist, the actual loading happens in process_query
if not st.session_state.embeddings_built:
    if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(TEXT_CHUNKS_PATH):
        # We don't load the FAISS object here, just confirm files exist and store path
        st.session_state.faiss_index_path = FAISS_INDEX_PATH
        st.session_state.all_chunks = load_chunks(TEXT_CHUNKS_PATH) # Load chunks once
        if st.session_state.all_chunks: # Check if chunks were successfully loaded
            st.session_state.embeddings_built = True
            st.sidebar.success("Loaded existing Knowledge Base (from disk).")
        else:
            st.session_state.faiss_index_path = None # Reset if chunks failed to load
            st.sidebar.warning("Could not load existing Knowledge Base chunks. Please rebuild it.")
    else:
        st.sidebar.info("No existing Knowledge Base found. Upload PDFs or place them in 'data/stories/' and click 'Rebuild Story Knowledge Base' to get started.")

# --- Chat Interface ---
# Display chat messages from history on app rerun
for message in st.session_state.messages:
    if message["role"] == "user":
        st.chat_message("user").write(message["content"])
    elif message["role"] == "assistant":
        with st.chat_message("assistant"):
            st.write(message["content"])
            # Only display image if image_url is not None
            if message.get("image_url"):
                st.image(message["image_url"], caption="Generated Image", use_column_width=True)

# Chat input
if query := st.chat_input("Ask me about Alice, Gulliver, or Arabian Nights..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": query})
    st.chat_message("user").write(query)

    # Process the query using the main orchestration logic
    response_data = process_query(
        query,
        st.session_state.faiss_index_path, # Pass the path instead of the object
        st.session_state.all_chunks,
        st.session_state.selected_tone,
        st.session_state.selected_embedding_model,
        st.session_state.selected_text_gen_model,
        st.session_state.selected_image_gen_model
    )

    story_response = response_data["story_response"]
    image_url = response_data["image_url"]

    # Add assistant response to chat history
    with st.chat_message("assistant"):
        st.write(story_response)
        # Only add image to history and display if image_url is not None
        if image_url:
            st.image(image_url, caption="Generated Image", use_column_width=True)
            st.session_state.messages.append({"role": "assistant", "content": story_response, "image_url": image_url})
        else:
            st.session_state.messages.append({"role": "assistant", "content": story_response})

# --- Clear Chat Button ---
if st.sidebar.button("Clear Chat"):
    st.session_state.messages = []
    st.rerun()

