# Whimsical Storyteller AI

This project implements a Retrieval Augmented Generation (RAG) based chatbot designed to answer user queries about classic public domain fictional stories: "Alice in Wonderland", "Gulliver's Travels", and "The Arabian Nights". The chatbot provides responses in a funny tone, generates related images, and gracefully handles irrelevant queries.

## Assignment Requirements Addressed

This project directly addresses all points outlined in the assignment:

1.  **Takes a query from the user:** Implemented via a `st.chat_input` widget in the Streamlit interface.
2.  **Replies to the user's query with a story from the training material in a funny tone:** Achieved through a RAG pipeline that retrieves relevant text chunks from the story PDFs and uses prompt engineering with an OpenAI LLM to generate responses in a user-selectable "funny" tone.
3.  **Outputs an image related to the story:** An image generation model (DALL-E) is used to create a relevant image based on the AI's story response, but only if the response is relevant to the knowledge base.
4.  **If the query is unrelated to the training material - replies with a "I don't know..." type message in a funny tone:** The RAG pipeline's retrieval mechanism naturally leads to "no relevant chunks" for out-of-scope queries. The LLM's system prompt is engineered to detect this and respond with a humorous "I don't know..." message, without generating an image.

## Evaluation Criteria Breakdown

This section details how the project's design and implementation meet the specified evaluation criteria:

1.  **Knowledge Training Logic:**
    * **Implementation:** Handled in `app/retriever.py` and `app/utils.py`.
    * **Process:** PDF files (either from the `data/stories/` directory or user uploads via Streamlit's `st.file_uploader`) are parsed using `PyPDF2`. The extracted text is then segmented into smaller, overlapping "chunks" to maintain context. For each chunk, a high-dimensional numerical representation (embedding) is generated using OpenAI's embedding models. These embeddings, along with their corresponding text chunks, are then stored in a FAISS (Facebook AI Similarity Search) index.
    * **Efficiency:** The process includes batching for embedding generation to comply with OpenAI API token limits, ensuring efficient processing of large documents. The FAISS index and chunks are persisted to disk, allowing for faster application restarts without re-embedding if the knowledge base hasn't changed.

2.  **Knowledge Retrieval Logic:**
    * **Implementation:** Primarily in `app/retriever.py`.
    * **Process:** When a user submits a query, an embedding is generated for that query using the *same* embedding model used for the story chunks. This query embedding is then used to perform a similarity search against the FAISS index. FAISS efficiently identifies and retrieves the top `k` (e.g., 3) most semantically similar text chunks from the stored stories.
    * **Robustness:** The FAISS index is reloaded from disk for each query, ensuring its integrity and preventing issues with Streamlit's session state serialization for complex C++-backed objects.

3.  **Output Tone Control:**
    * **Implementation:** Managed in `app/responder.py` and `streamlit_app.py`.
    * **Mechanism:** The Streamlit sidebar provides a dropdown for the user to select the desired output tone (e.g., "Funny", "Narrator", "Whimsical"). This selected tone is dynamically injected into the system prompt of the Large Language Model (LLM) before generating a response. The LLM is explicitly instructed to adopt this tone, ensuring consistent stylistic output.

4.  **Image Creation Logic:**
    * **Implementation:** Defined in `app/image_gen.py` and orchestrated in `app/main.py`.
    * **Process:** After the LLM generates a text response, a secondary prompt is sent to the LLM to extract concise, relevant keywords or a summary from that response. This extracted information then serves as the prompt for an OpenAI DALL-E image generation model (DALL-E 2 or DALL-E 3).
    * **Conditional Generation:** A crucial check is implemented: an image is *only* generated if the AI's text response is deemed relevant to the knowledge base. If the response is an "I don't know..." type message (indicating irrelevance), no image generation API call is made, saving API costs and improving user experience.

5.  **Ease of Changing Models:**
    * **Implementation:** Centralized in `app/config.py` and abstracted in `app/utils.py`.
    * **Flexibility:** All model names (for embeddings, text generation, and image generation) are defined in `app/config.py`. The `app/utils.py` module provides generic `get_embedding_model`, `get_llm_model`, and `get_image_model` functions that return the appropriate OpenAI client based on the selected model name.
    * **User Interface:** Streamlit's sidebar provides dropdowns, allowing users to switch between different OpenAI models at runtime without code changes.
    * **Future-Proofing:** The modular design facilitates integrating local open-source models (e.g., Sentence Transformers for embeddings, Llama-2/Mistral for text generation) by modifying only `app/config.py` and `app/utils.py` to include their respective client initializations.

6.  **Accuracy:**
    * **RAG's Core Benefit:** The RAG architecture fundamentally enhances accuracy by grounding the LLM's responses in the specific content retrieved from the story PDFs. This significantly reduces the likelihood of hallucinations (fabricated information).
    * **Prompt Engineering:** The LLM's system prompt explicitly instructs it to "MUST ONLY use the provided 'Context'" and to generate an "I don't know..." message if the context is insufficient or irrelevant.
    * **Irrelevance Handling:** The logic to detect irrelevant queries and provide a funny "I don't know..." response ensures that the chatbot accurately communicates its limitations when faced with out-of-scope questions.

7.  **Use of Open Source Technologies Wherever Possible:**
    * **Core Frameworks:** The project leverages several open-source Python libraries:
        * `streamlit`: For building the interactive web interface.
        * `PyPDF2`: For parsing PDF documents.
        * `tiktoken`: For OpenAI tokenization (though OpenAI-specific, it's open source).
        * `faiss-cpu`: For efficient similarity search (vector database).
        * `python-dotenv`: For local environment variable management.
        * `httpx`: The underlying HTTP client used by the OpenAI library.
    * **Model Choice:** As requested, OpenAI's proprietary models are used for LLM, embeddings, and image generation. However, the architecture is designed to allow for easy integration of open-source alternatives (e.g., `sentence-transformers` for embeddings, local LLMs like Llama-2/Mistral via `Hugging Face Transformers` or `ollama`) by modifying the `app/config.py` and `app/utils.py` files.

8.  **System Design:**
    * **Modularity:** The project is structured into logical Python modules (`app/main.py`, `app/retriever.py`, `app/responder.py`, `app/image_gen.py`, `app/utils.py`, `app/config.py`), each with a clear responsibility. This promotes code organization, reusability, and easier debugging.
    * **Separation of Concerns:** Data handling, retrieval, response generation, image creation, and configuration are all separated into distinct files.
    * **Persistence:** The FAISS index and text chunks are saved to disk, improving startup times and robustness across Streamlit reruns.
    * **Error Handling:** Comprehensive `try-except` blocks are used throughout the application to gracefully handle API errors, file loading issues, and other unexpected exceptions, providing informative messages to the user and console.
    * **Scalability:** The RAG approach and modular design make it relatively straightforward to expand the knowledge base (add more PDFs) or integrate new models in the future.

## Setup and Installation

1.  **Clone the repository (or create the files manually):**
    ```bash
    git clone <your-repo-url>
    cd <your-repo-name>
    ```

2.  **Create a Python Virtual Environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows:
    .\venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Place your PDF Story Files:**
    * Create the `data/stories/` directory if it doesn't exist.
    * Download your PDF files for "Alice in Wonderland", "Gulliver's Travels", and "The Arabian Nights" and place them inside the `data/stories/` directory. (Alternatively, you can upload them directly in the Streamlit app's sidebar).

5.  **Set up your OpenAI API Key:**
    * **For Local Development:** Create a file named `.env` in the root directory of your project (the same directory as `streamlit_app.py`). Add your OpenAI API key to this file:
        ```
        OPENAI_API_KEY="your_openai_api_key_here"
        ```
    * **For Streamlit Cloud Deployment:** In your Streamlit Cloud app settings, navigate to "Secrets" and add a new secret with the key `OPENAI_API_KEY` and your actual OpenAI API key as the value.
    * **Important:** Keep your API key secure and do not share it publicly.

## Running the Application

1.  **Activate your virtual environment (if you haven't already):**
    ```bash
    # On Windows:
    .\venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```

2.  **Run the Streamlit application:**
    ```bash
    streamlit run streamlit_app.py
    ```

3.  **Access the App:**
    * Your web browser will automatically open to the Streamlit application (usually `http://localhost:8501`).

## Usage

1.  **Build Knowledge Base:**
    * On the left sidebar, under "Knowledge Base", you can either upload your PDF story files directly or ensure they are placed in the `data/stories/` folder.
    * Click the "Rebuild Story Knowledge Base" button. This crucial step will parse your PDFs, create text chunks, generate embeddings, and store them in a FAISS index. It will save the index and chunks to the `embeddings/` directory for faster loading on subsequent app runs.

2.  **Select Tone and Models:**
    * Use the dropdowns in the sidebar to choose your desired output tone and the OpenAI models for embeddings, text generation, and image generation.

3.  **Chat:**
    * Type your query in the chat input box at the bottom of the page.
    * The AI will respond with a story-related answer in the chosen tone and generate an accompanying image (if the query is relevant).
    * If your query is unrelated to the stories, the AI will respond with a funny "I don't know..." message, and no image will be generated.

4.  **Clear Chat:**
    * Use the "Clear Chat" button in the sidebar to reset the conversation.
