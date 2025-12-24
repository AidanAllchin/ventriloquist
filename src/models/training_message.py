"""
File: models/training_message.py
Author: Aidan Allchin
Created: 2025-12-23
Last Modified: 2025-12-23
"""

from pydantic import BaseModel, Field
from typing import List, Optional


class TrainingMessage(BaseModel):
    """Lightweight message model for training data."""

    chat_id: str = Field(..., description="Unique identifier for the conversation")
    from_contact: str = Field(..., description="Name of contact who sent the message")
    timestamp: str = Field(..., description="UTC ISO timestamp")
    content: str = Field(..., description="Message body")
    is_group_chat: bool = Field(..., description="Whether this is a group chat message")
    chat_members: List[str] = Field(default_factory=list, description="List of all contact names in the chat")
    reply_to_text: Optional[str] = Field(None, description="Text of the message being replied to (truncated)")
    thread_originator_guid: Optional[str] = Field(None, description="GUID of original message in thread (for out-of-window lookups)")
