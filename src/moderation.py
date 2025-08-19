from __future__ import annotations
import os
from typing import Tuple
from openai import OpenAI

_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))

def moderate_text(text: str) -> Tuple[bool, str]:
    """
    Use OpenAI moderation to decide if text is allowed.
    Model: omni-moderation-latest
    """
    resp = _client.moderations.create(
        model="omni-moderation-latest",
        input=text
    )
    result = resp.results[0]
    flagged = result.flagged
    reason = "Flagged by moderation policy." if flagged else "Allowed."
    return (not flagged), reason
