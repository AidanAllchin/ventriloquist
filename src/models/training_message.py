"""
File: models/training_message.py
Author: Aidan Allchin
Created: 2025-12-23
Last Modified: 2025-12-27
"""

import json
from datetime import timedelta
from pydantic import BaseModel, Field
from typing import List, Optional


# Time delta bucket thresholds (upper bounds)
DELTA_BUCKETS = [
    (timedelta(minutes=1), "<1m"),
    (timedelta(minutes=5), "<5m"),
    (timedelta(hours=1), "<1h"),
    (timedelta(hours=12), "<12h"),
    (timedelta(days=1), "<1d"),
]


def compute_delta_bucket(delta: timedelta) -> str:
    """
    Convert a timedelta to a bucket label.

    Args:
        delta: Time difference from previous message

    Returns:
        Bucket label like "<1m", "<5m", "<1h", "<12h", "<1d", or "1d+"
    """
    for threshold, label in DELTA_BUCKETS:
        if delta < threshold:
            return label
    return "1d+"


class TrainingMessage(BaseModel):
    """Lightweight message model for training data."""

    chat_id: str = Field(..., description="Unique identifier for the conversation")
    from_contact: str = Field(..., description="Name of contact who sent the message")
    timestamp: str = Field(..., description="UTC ISO timestamp")
    content: str = Field(..., description="Message body or attachment description")
    content_type: str = Field("text", description="Content type: text, image, video, audio, document, file, location, contact")
    is_group_chat: bool = Field(..., description="Whether this is a group chat message")
    chat_members: List[str] = Field(default_factory=list, description="List of all contact names in the chat")
    reply_to_text: Optional[str] = Field(None, description="Text of the message being replied to (truncated)")
    thread_originator_guid: Optional[str] = Field(None, description="GUID of original message in thread (for out-of-window lookups)")

    def to_window_json(self, delta_bucket: str) -> str:
        """
        Format this message as a JSON line for training windows.

        Args:
            delta_bucket: Pre-computed time delta bucket (e.g., "<5m", "<1h")

        Returns:
            JSON string: {"name": "...", "delta": "...", "reply_to": ..., "content_type": "...", "text": "..."}
        """
        return json.dumps(
            {
                "name": self.from_contact,
                "delta": delta_bucket,
                "reply_to": self.reply_to_text,  # null or truncated string
                "content_type": self.content_type,
                "text": self.content,
            },
            ensure_ascii=False,
        )
