from app.utils import get_llm_model, get_image_model
from openai import APIError

def generate_image_prompt(story_response: str, llm_model_name: str) -> str:
    
    llm_client = get_llm_model(llm_model_name)

    system_prompt = """
    You are an expert image prompt generator. Your task is to create a concise, vivid, and imaginative
    prompt for an AI image generation model (like DALL-E) based on the provided story response.
    Focus on key characters, settings, and actions described.
    The prompt should be suitable for generating a whimsical, fantastical, or classic illustration style.
    Keep the prompt to a maximum of 50 words.
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Story Response: {story_response}\n\nImage Prompt:"}
    ]

    try:
        response = llm_client.create(
            model=llm_model_name,
            messages=messages,
            max_tokens=50, 
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except APIError as e:
        print(f"OpenAI API Error generating image prompt: {e}")
        return "A fantastical scene from a storybook." 
    except Exception as e:
        print(f"An unexpected error occurred generating image prompt: {e}")
        return "A whimsical illustration." 

def generate_image(image_prompt: str, image_model_name: str) -> str:
    
    image_client = get_image_model(image_model_name)

    try:
        if image_model_name == "dall-e-3":
            response = image_client.generate(
                model="dall-e-3",
                prompt=image_prompt,
                size="1024x1024",
                quality="standard",
                n=1
            )
        elif image_model_name == "dall-e-2":
            response = image_client.generate(
                model="dall-e-2",
                prompt=image_prompt,
                size="512x512", 
                n=1
            )
        else:
            raise ValueError(f"Unsupported image model: {image_model_name}")

        return response.data[0].url
    except APIError as e:
        print(f"OpenAI API Error generating image: {e}")
        return f"https://placehold.co/512x512/FF0000/FFFFFF?text=Image+Error%3A+{e.code}" 
    except Exception as e:
        print(f"An unexpected error occurred generating image: {e}")
        return "https://placehold.co/512x512/0000FF/FFFFFF?text=Image+Gen+Failed" 
