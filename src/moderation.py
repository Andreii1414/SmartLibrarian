from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Optional, Tuple

from openai import OpenAI
from openai import APIError


# -------------------------
# Data models
# -------------------------
@dataclass(frozen=True)
class ModerationConfig:
    api_key: str
    model: str = "omni-moderation-latest"


@dataclass(frozen=True)
class ModerationResult:
    allowed: bool
    reason: str
    raw: Optional[object] = None


# -------------------------
# Client
# -------------------------
class ModerationClient:
    """
    Client for OpenAI moderation API.
    """

    def __init__(self, config: Optional[ModerationConfig] = None):
        api_key = os.getenv("OPENAI_API_KEY", "") if config is None else config.api_key
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required for moderation.")
        self.model = config.model if config else "omni-moderation-latest"
        self._client = OpenAI(api_key=api_key)

    def check(self, text: str) -> ModerationResult:
        """
        Run moderation on a single text string.
        Returns ModerationResult(allowed, reason, raw)
        """
        try:
            resp = self._client.moderations.create(model=self.model, input=text)
            # SDK returns .results list; flagged True when any policy is triggered
            result = resp.results[0]
            flagged = bool(result.flagged)
            reason = "Allowed." if not flagged else "Flagged by moderation policy."
            return ModerationResult(allowed=not flagged, reason=reason, raw=result)
        except APIError as e:
            # Fail-safe: block on API errors to avoid sending unmoderated content
            return ModerationResult(
                allowed=False, reason=f"Moderation service error: {e}", raw=None
            )
        except Exception as e:
            return ModerationResult(
                allowed=False, reason=f"Moderation unexpected error: {e}", raw=None
            )


# -------------------------
# Backward-compatible wrapper
# -------------------------
_client_singleton: Optional[ModerationClient] = None


def _get_client() -> ModerationClient:
    """
    Get the singleton ModerationClient instance.
    """
    global _client_singleton
    if _client_singleton is None:
        _client_singleton = ModerationClient()
    return _client_singleton


def moderate_text(text: str) -> Tuple[bool, str]:
    """
    Backward-compatible function:
    returns (allowed: bool, reason: str)
    """
    res = _get_client().check(text)
    return (res.allowed, res.reason)
