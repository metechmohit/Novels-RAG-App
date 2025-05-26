import os
import json
from pypdf import PdfReader
from openai import OpenAI, APIError 
import tiktoken
from typing import List, Dict
import io
import streamlit as st
import httpx 

# --- Explicitly create an httpx client ignoring environment variables ---
# This prevents httpx from automatically picking up HTTP_PROXY/HTTPS_PROXY
# which can cause conflicts with how the OpenAI client expects arguments.
default_http_client = None
try:
    # Set trust_env=False to explicitly tell httpx to ignore environment variables
    # related to proxies, certificates, etc.
    default_http_client = httpx.Client(trust_env=False)
except Exception as e:
    print(f"Warning: Could not create httpx.Client with trust_env=False: {e}. Falling back to default client.")


# Initialize OpenAI client globally using the custom httpx client
# Pass http_client=None if default_http_client could not be initialized
client = None
try:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), http_client=default_http_client)
except APIError as e:
    print(f"CRITICAL ERROR: Failed to initialize OpenAI client due to API error: {e}. Please check your API key and network connection.")
    # In a real application, you might want to log this more robustly or exit.
    # For Streamlit, the error will propagate and be shown in the UI.
except Exception as e:
    print(f"CRITICAL ERROR: An unexpected error occurred during OpenAI client initialization: {e}.")
    # For Streamlit, the error will propagate and be shown in the UI.


def load_pdfs(directory: str) -> List[Dict[str, str]]:
    
    stories = []
    if not os.path.exists(directory):
        print(f"Directory not found: {directory}")
        return stories

    for filename in os.listdir(directory):
        if filename.endswith(".pdf"):
            filepath = os.path.join(directory, filename)
            try:
                reader = PdfReader(filepath)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() or ""
                stories.append({"title": filename.replace(".pdf", ""), "content": text})
                print(f"Successfully loaded {filename} from file system.")
            except Exception as e:
                print(f"Error loading PDF {filename} from file system: {e}")
    return stories

def load_uploaded_pdfs(uploaded_files: List[st.runtime.uploaded_file_manager.UploadedFile]) -> List[Dict[str, str]]:
    
    stories = []
    if not uploaded_files:
        return stories

    for uploaded_file in uploaded_files:
        try:
            # Read the uploaded file as a BytesIO object
            pdf_bytes = io.BytesIO(uploaded_file.getvalue())
            reader = PdfReader(pdf_bytes)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
            stories.append({"title": uploaded_file.name.replace(".pdf", ""), "content": text})
            print(f"Successfully loaded {uploaded_file.name} from upload.")
        except Exception as e:
            print(f"Error loading uploaded PDF {uploaded_file.name}: {e}")
    return stories


def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    
    if not text:
        return []

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - chunk_overlap
        if start >= len(text) - chunk_overlap: # Ensure last chunk is not too small
            break
    return chunks

def get_embedding_model(model_name: str):
   
    if client is None:
        raise RuntimeError("OpenAI client not initialized. Check API key and network.")
    return client.embeddings


def get_llm_model(model_name: str):
   
    if client is None:
        raise RuntimeError("OpenAI client not initialized. Check API key and network.")
    return client.chat.completions

def get_image_model(model_name: str):
    
    if client is None:
        raise RuntimeError("OpenAI client not initialized. Check API key and network.")
    return client.images

def num_tokens_from_string(string: str, model_name: str) -> int:
    
    try:
        encoding = tiktoken.encoding_for_model(model_name)
    except KeyError:
        # Fallback for models not explicitly in tiktoken's registry
        encoding = tiktoken.get_encoding("cl100k_base")
    num_tokens = len(encoding.encode(string))
    return num_tokens

def save_chunks(chunks: List[str], path: str):
    """Saves text chunks to a JSON file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, ensure_ascii=False, indent=4)

def load_chunks(path: str) -> List[str]:
    """Loads text chunks from a JSON file."""
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []
