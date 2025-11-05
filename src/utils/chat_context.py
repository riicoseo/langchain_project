"""
Utility helpers for building lightweight conversation context strings.
"""

from __future__ import annotations

import json
from typing import Any, Sequence


def build_conversation_context(messages: Sequence[Any], max_turns: int = 3) -> str:
    """Return a short text block summarising recent conversation history.

    The summary is intended to help downstream prompts resolve pronouns or hints
    without re-injecting the full LangChain message history.
    """
    if not messages:
        return ""

    history = list(messages)
    if history:
        # Exclude the latest message, which is the current user query.
        history = history[:-1]

    if not history:
        return ""

    # Keep only the most recent turns (user+assistant pairs).
    window = history[-max_turns * 2 :]
    lines = []

    for msg in window:
        role = getattr(msg, "type", getattr(msg, "role", "")).lower()
        if role == "human":
            role_label = "User"
        elif role == "ai":
            role_label = "Assistant"
        elif role == "system":
            role_label = "System"
        else:
            role_label = role.capitalize() if role else "Message"

        content = getattr(msg, "content", "")
        if isinstance(content, list):
            # LangChain sometimes stores content as a list of chunk dicts.
            content = " ".join(
                str(chunk.get("text", chunk)) if isinstance(chunk, dict) else str(chunk)
                for chunk in content
                if chunk
            )
        elif isinstance(content, dict):
            content = json.dumps(content, ensure_ascii=False)

        text = str(content).strip()
        if not text:
            continue

        lines.append(f"{role_label}: {text}")

    if not lines:
        return ""

    return "Recent conversation context:\n" + "\n".join(lines)
