from typing import List, Tuple
from app.utils import get_llm_model, num_tokens_from_string
from openai import APIError

def generate_response(
    query: str,
    relevant_chunks: List[str],
    tone: str,
    llm_model_name: str
) -> Tuple[str, bool]: 
    llm_client = get_llm_model(llm_model_name)

    # Determine if there's relevant context
    has_relevant_context = bool(relevant_chunks)

    # Construct the system prompt for tone control and instruction
    system_prompt = f"""
    You are a whimsical storyteller who loves to share tales from classic public domain literature like "Alice in Wonderland", "Gulliver's Travels", and "The Arabian Nights".
    Your primary goal is to answer the user's questions about these stories in a {tone.lower()} tone.
    You MUST ONLY use the provided 'Context' below to formulate your answer.
    If the provided 'Context' is empty or does not contain enough information to answer the query, or if the query is completely unrelated to the stories,
    you MUST reply with a funny, {tone.lower()} "I don't know..." type message, admitting you can't find the answer in your storybooks.
    Keep your responses concise and engaging.
    """

    # Combine relevant chunks into a single context string
    context_str = "\n\n".join(relevant_chunks)
    if not context_str:
        context_str = "No relevant story context found." # Fallback if chunks are empty

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Context:\n{context_str}\n\nUser Query: {query}"}
    ]

    response_text = ""
    is_relevant_response = has_relevant_context # Start with assumption based on retrieved chunks

    try:
        max_tokens_for_response = 500 # Max tokens for the LLM's answer
        
        response = llm_client.create(
            model=llm_model_name,
            messages=messages,
            max_tokens=max_tokens_for_response,
            temperature=0.7, # A bit of creativity for funny tone
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0
        )
        response_text = response.choices[0].message.content.strip()

        lower_response = response_text.lower()
        if not has_relevant_context or \
           "i don't know" in lower_response or \
           "can't find that in my storybooks" in lower_response or \
           "my storybook seems to have a few missing pages" in lower_response or \
           "my quill snapped" in lower_response or \
           "oopsie" in lower_response or \
           "oh dear" in lower_response:
            is_relevant_response = False
        else:
            is_relevant_response = True # If chunks were there and LLM didn't use an "I don't know" phrase

        return response_text, is_relevant_response

    except APIError as e:
        print(f"OpenAI API Error in responder: {e}")
        return f"Oh dear! My storybook seems to have a few missing pages right now. I encountered an error: {e.code}. Perhaps try a different question, or check my magical connection!", False
    except Exception as e:
        print(f"An unexpected error occurred in responder: {e}")
        return "Oopsie! My quill snapped while trying to write that response. Something went unexpectedly wrong. Try again!", False
