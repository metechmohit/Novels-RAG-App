import streamlit as st
import faiss # Import faiss here
from app.retriever import retrieve_relevant_chunks, load_faiss_index_and_chunks # Import load_faiss_index_and_chunks
from app.responder import generate_response
from app.image_gen import generate_image_prompt, generate_image
from app.config import DEFAULT_EMBEDDING_MODEL, DEFAULT_TEXT_GENERATION_MODEL, DEFAULT_IMAGE_GENERATION_MODEL, FAISS_INDEX_PATH, TEXT_CHUNKS_PATH # Import paths

def process_query(
    query: str,
    faiss_index_path: str, 
    all_chunks,  
    selected_tone: str,
    embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
    text_gen_model_name: str = DEFAULT_TEXT_GENERATION_MODEL,
    image_gen_model_name: str = DEFAULT_IMAGE_GENERATION_MODEL
) -> dict:
    
    # Load FAISS index and chunks on each query to ensure fresh state
    current_faiss_index = None
    current_all_chunks = []

    if faiss_index_path and all_chunks: # Check if path and chunks are available in session state
        try:
            current_faiss_index = faiss.read_index(faiss_index_path)
            current_all_chunks = all_chunks # Chunks are already in session state
            print("FAISS index successfully loaded from disk for current query.")
        except Exception as e:
            st.error(f"Error loading FAISS index from disk: {e}. Please try rebuilding the knowledge base.")
            print(f"Error loading FAISS index from disk: {e}")
            return {
                "story_response": "Oops! My memory seems to have a glitch. I couldn't load my story index. Please try rebuilding the knowledge base!",
                "image_url": None
            }
    else:
        return {
            "story_response": "My storybooks are currently empty! Please ensure the PDF files are in 'data/stories/' or uploaded, and try rebuilding the knowledge base.",
            "image_url": None
        }

    # 1. Retrieve relevant chunks
    with st.spinner("Searching through my storybooks..."):
        relevant_chunks = retrieve_relevant_chunks(
            query,
            current_faiss_index, # Use the freshly loaded index
            current_all_chunks,  # Use the chunks from session state
            embedding_model_name
        )
        if not relevant_chunks:
            print("No relevant chunks found for the query, LLM will generate 'I don't know' response.")

    # 2. Generate story response and get relevance flag
    with st.spinner("Crafting a response with a touch of magic..."):
        story_response, is_relevant_for_image = generate_response( # Capture the relevance flag
            query,
            relevant_chunks,
            selected_tone,
            text_gen_model_name
        )

    image_url = None # Initialize image_url to None by default

    # 3. Conditionally generate image based on relevance
    if is_relevant_for_image:
        with st.spinner("Dreaming up an image..."):
            image_prompt = generate_image_prompt(story_response, text_gen_model_name)
        
        with st.spinner("Painting a picture for you..."):
            image_url = generate_image(image_prompt, image_gen_model_name)
    else:
        print("Query not relevant or LLM generated 'I don't know' response. Skipping image generation.")
        # image_url remains None, so no image will be displayed

    return {
        "story_response": story_response,
        "image_url": image_url
    }
