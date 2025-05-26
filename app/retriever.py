import os
import numpy as np
import faiss
from typing import List, Tuple
from app.utils import load_pdfs, load_uploaded_pdfs, chunk_text, get_embedding_model, save_chunks, load_chunks, num_tokens_from_string
from app.config import DATA_DIR, EMBEDDINGS_DIR, FAISS_INDEX_PATH, TEXT_CHUNKS_PATH
import streamlit as st 

def create_and_store_embeddings(
    embedding_model_name: str,
    uploaded_files: List[st.runtime.uploaded_file_manager.UploadedFile] = None, 
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> Tuple[faiss.Index, List[str]]:
    
    print(f"Starting embedding creation with model: {embedding_model_name}")

    stories = []
    if uploaded_files:
        stories = load_uploaded_pdfs(uploaded_files)
        if not stories:
            print("No valid PDF stories found in uploaded files.")
    
    if not stories: # Fallback to file system if no uploaded files or no valid uploaded files
        print(f"Attempting to load PDFs from file system: {DATA_DIR}")
        stories = load_pdfs(DATA_DIR)
        if not stories:
            print("No PDF stories found in 'data/stories/' to process.")
            return None, []

    all_chunks = []
    for story in stories:
        story_chunks = chunk_text(story["content"], chunk_size, chunk_overlap)
        all_chunks.extend(story_chunks)

    if not all_chunks:
        print("No chunks generated from stories.")
        return None, []

    print(f"Generated {len(all_chunks)} chunks.")

    # Get the embedding client
    embedding_client = get_embedding_model(embedding_model_name)
    embeddings = []
    embedding_dimension = 0

    BATCH_SIZE = 500 

    for i in range(0, len(all_chunks), BATCH_SIZE):
        batch = all_chunks[i:i + BATCH_SIZE]
        print(f"Processing batch {i // BATCH_SIZE + 1}/{(len(all_chunks) + BATCH_SIZE - 1) // BATCH_SIZE} with {len(batch)} chunks...")
        try:
            response = embedding_client.create(
                input=batch,
                model=embedding_model_name
            )
            batch_embeddings = [item.embedding for item in response.data]
            embeddings.extend(batch_embeddings)
            if not embedding_dimension and batch_embeddings:
                embedding_dimension = len(batch_embeddings[0])

        except Exception as e:
            print(f"Error generating embeddings for batch starting at index {i}: {e}")
            st.error(f"Error generating embeddings for a batch. Please check your OpenAI API usage and limits. Error: {e}")
            return None, [] # Stop processing if a batch fails

    print(f"Generated {len(embeddings)} embeddings in total with dimension {embedding_dimension}.")

    if not embeddings:
        print("Failed to generate any embeddings.")
        return None, []

    # Convert embeddings to numpy array
    embeddings_np = np.array(embeddings).astype('float32')

    # Create FAISS index
    index = faiss.IndexFlatL2(embedding_dimension) # L2 distance for similarity
    index.add(embeddings_np)
    print(f"FAISS index created with {index.ntotal} vectors.")

    # Save FAISS index and chunks
    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
    faiss.write_index(index, FAISS_INDEX_PATH)
    save_chunks(all_chunks, TEXT_CHUNKS_PATH)
    print(f"FAISS index saved to {FAISS_INDEX_PATH}")
    print(f"Text chunks saved to {TEXT_CHUNKS_PATH}")

    return index, all_chunks

def load_faiss_index_and_chunks() -> Tuple[faiss.Index, List[str]]:
    if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(TEXT_CHUNKS_PATH):
        try:
            index = faiss.read_index(FAISS_INDEX_PATH)
            chunks = load_chunks(TEXT_CHUNKS_PATH)
            print(f"Loaded FAISS index with {index.ntotal} vectors and {len(chunks)} chunks.")
            return index, chunks
        except Exception as e:
            print(f"Error loading FAISS index or chunks: {e}")
            return None, []
    return None, []

def retrieve_relevant_chunks(
    query: str,
    index: faiss.Index,
    all_chunks: List[str],
    embedding_model_name: str,
    top_k: int = 3
) -> List[str]:
    
    if not query or not index or not all_chunks:
        return []

    embedding_client = get_embedding_model(embedding_model_name)
    try:
        # Generate embedding for the query
        query_embedding_response = embedding_client.create(
            input=[query],
            model=embedding_model_name
        )
        query_embedding = np.array(query_embedding_response.data[0].embedding).astype('float32').reshape(1, -1)

        # Perform similarity search
        distances, indices = index.search(query_embedding, top_k)

        relevant_chunks = [all_chunks[i] for i in indices[0] if i < len(all_chunks)]
        print(f"Retrieved {len(relevant_chunks)} relevant chunks.")
        return relevant_chunks

    except Exception as e:
        print(f"Error retrieving relevant chunks: {e}")
        return []
