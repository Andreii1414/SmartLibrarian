from __future__ import annotations
import os
from openai import OpenAI

_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))

def generate_book_image(title: str, summary: str, size: str = "512x512") -> str:
    """
    Generate an image prompt based on the book title + summary and return image URL.

    Args:
        title: Book title
        summary: Short or full summary of the book
        size: "256x256", "512x512", "1024x1024"

    Returns:
        URL of the generated image (string)
    """
    prompt = (
        f"Create a detailed, artistic illustration that could serve as a book cover. "
        f"Book title: '{title}'. "
        "The image should be imaginative, evocative, and thematically representative of the story."
    )

    resp = _client.images.generate(
        model="dall-e-3",
        prompt=prompt,
        size=size,
    )

    return resp.data[0].url
