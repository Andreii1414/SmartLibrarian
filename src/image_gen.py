from __future__ import annotations
import os
from openai import OpenAI

_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))

def generate_book_image(title: str, size: str = "1024x1024") -> str:
    """
    Generate a book cover image using OpenAI's DALL-E model.
    """
    prompt = (
        f"Create a book cover image for a book titled '{title}'. "
        f"Do not include any text or logo on the image. "
        "The image should be imaginative, evocative, thematically representative of the story, but without too much detail."
    )

    resp = _client.images.generate(
        model="dall-e-3",
        prompt=prompt,
        size=size,
    )

    return resp.data[0].url
