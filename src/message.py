import openai
from openai import OpenAI

def get_response( api_key: str, system_prompt: str, user_prompt: str) -> str:
    """
    Get a response from the OpenAI API based on the provided prompt.
    
    Args:
        prompt (str): The prompt to send to the OpenAI API.
        
    Returns:
        str: The response from the OpenAI API.
    """

    client = OpenAI(api_key=api_key)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    return response.choices[0].message.content
