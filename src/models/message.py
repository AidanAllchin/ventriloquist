"""
Message record model.

File: models/message.py
Author: Aidan Allchin
Created: 2025-11-23
Last Modified: 2025-12-27
"""

from datetime import datetime
from typing import List, Dict, Optional, Any

from pydantic import BaseModel, Field, ConfigDict
from rich.text import Text

from .attachment import AttachmentInfo


class MessageRecord(BaseModel):
    """Represents a message to be stored in the database"""
    model_config = ConfigDict(
        json_encoders={datetime: lambda v: v.isoformat()},
        validate_assignment=True,
        use_enum_values=True,
        extra='ignore'
    )

    message_id: int = Field(..., description="message.ROWID from iMessage DB (device-specific, auto-increment)", gt=0)
    guid: str = Field(..., description="message.guid - globally unique UUID for cross-device sync", min_length=1)
    text: Optional[str] = Field(None, description="Message text content")
    timestamp: datetime = Field(..., description="When the message was sent")
    sender_id: Optional[str] = Field(None, description="phone/email of sender (NULL for sent messages)")
    recipient_id: Optional[str] = Field(None, description="phone/email of recipient (your number for received messages)")
    is_from_me: bool = Field(..., description="Already determined by iMessage DB")
    service: str = Field(..., description="iMessage, SMS, or RCS", min_length=1)
    chat_identifier: Optional[str] = Field(None, description="Internal chat ID (hash for groups, phone for individual)")
    is_group_chat: bool = Field(False, description="True for group chats")
    group_chat_name: Optional[str] = Field(None, description="User-visible chat name (can be NULL)")
    group_chat_participants: Optional[List[str]] = Field(None, description="List of all participant phone/email IDs in group chat")
    has_attachments: bool = Field(False, description="True if message has attachments")
    is_audio_message: bool = Field(False, description="True for voice memos (set by iMessage)")
    attachments: Optional[List[AttachmentInfo]] = Field(None, description="List of attachment metadata")
    is_read: bool = Field(False, description="Read status")
    date_read: Optional[datetime] = Field(None, description="When message was read")
    date_delivered: Optional[datetime] = Field(None, description="When message was delivered")
    reply_to_guid: Optional[str] = Field(None, description="GUID of message this replies to")
    thread_originator_guid: Optional[str] = Field(None, description="GUID of original message in thread")

    def to_db_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database insertion"""
        return {
            "message_id": self.message_id,
            "guid": self.guid,
            "text": self.text,
            "timestamp": self.timestamp.isoformat(),
            "sender_id": self.sender_id,
            "recipient_id": self.recipient_id,
            "is_from_me": self.is_from_me,
            "service": self.service,
            "chat_identifier": self.chat_identifier,
            "is_group_chat": self.is_group_chat,
            "group_chat_name": self.group_chat_name,
            "group_chat_participants": self.group_chat_participants,
            "has_attachments": self.has_attachments,
            "is_read": self.is_read,
            "read_timestamp": self.date_read.isoformat() if self.date_read else None,
            "delivered_timestamp": self.date_delivered.isoformat() if self.date_delivered else None,
            "reply_to_guid": self.reply_to_guid,
            "thread_originator_guid": self.thread_originator_guid,
        }

    @classmethod
    def from_db_dict(cls, data: Dict[str, Any]) -> "MessageRecord":
        """Create MessageRecord instance from database dictionary"""
        import json

        # Parse timestamp strings
        if isinstance(data.get("timestamp"), str):
            data["timestamp"] = datetime.fromisoformat(data["timestamp"])

        if isinstance(data.get("read_timestamp"), str):
            data["date_read"] = datetime.fromisoformat(data["read_timestamp"])
        elif "read_timestamp" in data:
            data["date_read"] = data["read_timestamp"]

        if isinstance(data.get("delivered_timestamp"), str):
            data["date_delivered"] = datetime.fromisoformat(data["delivered_timestamp"])
        elif "delivered_timestamp" in data:
            data["date_delivered"] = data["delivered_timestamp"]

        # Parse JSON string for group_chat_participants
        if isinstance(data.get("group_chat_participants"), str):
            data["group_chat_participants"] = json.loads(data["group_chat_participants"])

        # Parse JSON string for attachments
        if isinstance(data.get("attachments"), str):
            attachments_data = json.loads(data["attachments"])
            data["attachments"] = [AttachmentInfo(**a) for a in attachments_data]
        elif data.get("attachments") is None:
            data["attachments"] = None

        # Convert integer booleans to actual booleans
        for bool_field in ["is_from_me", "is_group_chat", "has_attachments", "is_audio_message", "is_read"]:
            if bool_field in data and isinstance(data[bool_field], int):
                data[bool_field] = bool(data[bool_field])

        return cls(**data)

    def to_rich_text(self) -> Text:
        """Format message as Rich Text for display"""
        if self.is_from_me:
            color = "cyan"
            direction = "→"
        else:
            color = "green"
            direction = "←"

        time_str = self.timestamp.strftime("%Y-%m-%d %H:%M:%S")

        text = Text()
        text.append(f"{direction} ", style=f"bold {color}")
        text.append(f"[{time_str}] ", style="dim")

        if self.is_from_me:
            text.append("You", style=f"bold {color}")
            text.append(" → ", style="dim")
            text.append(f"{self.recipient_id or 'Unknown'}", style=f"bold {color}")
        else:
            text.append(f"{self.sender_id or 'Unknown'}", style=f"bold {color}")
            text.append(" → ", style="dim")
            text.append("You", style=f"bold {color}")

        if self.is_group_chat and self.group_chat_name:
            text.append(f" ({self.group_chat_name})", style="yellow")
            if self.group_chat_participants:
                text.append(f" [{len(self.group_chat_participants)} participants]", style="dim yellow")

        text.append("\n  ")

        if self.text and self.text.strip():
            content = self.text[:200] + "..." if len(self.text) > 200 else self.text
            text.append(content, style="white")
        elif self.has_attachments:
            text.append("[Attachment]", style="blue italic")
        else:
            text.append("[Empty message]", style="dim italic")

        indicators = []
        if self.has_attachments and self.text:
            indicators.append("[A]")
        if self.reply_to_guid:
            indicators.append("[R]")
        if self.is_read and not self.is_from_me:
            indicators.append("[Read]")

        if indicators:
            text.append(f" {' '.join(indicators)}", style="dim")

        return text
