from __future__ import annotations
import os
from openai import OpenAI

_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))

def generate_book_image(title: str, summary: str, size: str = "1024x1024") -> str:
    """
    Generate a book cover image using OpenAI's DALL-E model.
    """
    prompt = (
        f"Create a detailed, artistic illustration that could serve as a book cover. "
        f"Book title: '{title}'. "
        f"Summary: '{" ".join(summary.split()[:100])}'. "
        f"Do not include any text on the image. "
        "The image should be imaginative, evocative, and thematically representative of the story."
    )

    resp = _client.images.generate(
        model="dall-e-3",
        prompt=prompt,
        size=size,
    )

    return resp.data[0].url
