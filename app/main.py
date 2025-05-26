# app/main.py

import streamlit as st
from app.retriever import retrieve_relevant_chunks
from app.responder import generate_response
from app.image_gen import generate_image_prompt, generate_image
from app.config import DEFAULT_EMBEDDING_MODEL, DEFAULT_TEXT_GENERATION_MODEL, DEFAULT_IMAGE_GENERATION_MODEL

def process_query(
    query: str,
    faiss_index, # Pass the FAISS index object
    all_chunks,  # Pass the list of all chunks
    selected_tone: str,
    embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
    text_gen_model_name: str = DEFAULT_TEXT_GENERATION_MODEL,
    image_gen_model_name: str = DEFAULT_IMAGE_GENERATION_MODEL
) -> dict:
    """
    Orchestrates the query processing, response generation, and image creation.
    Generates an image only if the response is relevant to the knowledge base.
    No image is generated or displayed for irrelevant queries.

    Args:
        query (str): The user's input query.
        faiss_index: The loaded FAISS index.
        all_chunks: The list of all text chunks.
        selected_tone (str): The tone selected by the user.
        embedding_model_name (str): Name of the embedding model.
        text_gen_model_name (str): Name of the text generation model.
        image_gen_model_name (str): Name of the image generation model.

    Returns:
        dict: A dictionary containing the generated story response and image URL (or None if no image).
    """
    # No default placeholder needed if we return None
    # NO_IMAGE_PLACEHOLDER = "https://placehold.co/512x512/808080/FFFFFF?text=No+Image+Available"

    if not faiss_index or not all_chunks:
        return {
            "story_response": "My storybooks are currently empty! Please ensure the PDF files are in 'data/stories/' or uploaded, and try rebuilding the knowledge base.",
            "image_url": None # Return None for image_url if no knowledge base
        }

    # 1. Retrieve relevant chunks
    with st.spinner("Searching through my storybooks..."):
        relevant_chunks = retrieve_relevant_chunks(
            query,
            faiss_index,
            all_chunks,
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
