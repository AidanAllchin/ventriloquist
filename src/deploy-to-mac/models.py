"""
Standalone data models for the iMessage logger.

This module contains Pydantic models used by the iMessage logger that runs on the Mac.

File: deploy-to-mac/models.py
Author: Aidan Allchin
Created: 2025-11-23
Last Modified: 2025-12-23
"""

from datetime import datetime, timezone
from typing import List, Dict, Optional, Any

from pydantic import BaseModel, Field, ConfigDict
from rich.text import Text


class MessageRecord(BaseModel):
    """Represents a message to be stored in the database"""
    model_config = ConfigDict(
        # Allow datetime objects to be serialized
        json_encoders={datetime: lambda v: v.isoformat()},
        # Validate assignment
        validate_assignment=True,
        # Use enum values
        use_enum_values=True,
        # Allow extra fields (for future extensibility)
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

    def to_rich_text(self) -> Text:
        """Format message as Rich Text for display"""
        # Color based on direction
        if self.is_from_me:
            color = "cyan"
            direction = "→"
        else:
            color = "green"
            direction = "←"

        # Format timestamp
        time_str = self.timestamp.strftime("%Y-%m-%d %H:%M:%S")

        # Build message text
        text = Text()
        text.append(f"{direction} ", style=f"bold {color}")
        text.append(f"[{time_str}] ", style="dim")

        # Contact info
        if self.is_from_me:
            text.append(f"You", style=f"bold {color}")
            text.append(" → ", style="dim")
            text.append(f"{self.recipient_id or 'Unknown'}", style=f"bold {color}")
        else:
            text.append(f"{self.sender_id or 'Unknown'}", style=f"bold {color}")
            text.append(" → ", style="dim")
            text.append("You", style=f"bold {color}")

        # Group chat indicator
        if self.is_group_chat and self.group_chat_name:
            text.append(f" ({self.group_chat_name})", style="yellow")
            if self.group_chat_participants:
                text.append(f" [{len(self.group_chat_participants)} participants]", style="dim yellow")

        text.append("\n  ")

        # Message content
        if self.text and self.text.strip():
            # Truncate long messages
            content = self.text[:200] + "..." if len(self.text) > 200 else self.text
            text.append(content, style="white")
        elif self.has_attachments:
            text.append("📎 [Attachment]", style="blue italic")
        else:
            text.append("[Empty message]", style="dim italic")

        # Add metadata indicators
        indicators = []
        if self.has_attachments and self.text:
            indicators.append("📎")
        if self.reply_to_guid:
            indicators.append("↩️")
        if self.is_read and not self.is_from_me:
            indicators.append("✓")

        if indicators:
            text.append(f" {' '.join(indicators)}", style="dim")

        return text


class MessageStatsRecord(BaseModel):
    """
    Aggregated message statistics for a specific identifier (phone/email).

    These statistics are computed and upserted after each batch of messages
    during the iMessage logging process.
    """
    model_config = ConfigDict(
        # Allow datetime objects to be serialized
        json_encoders={datetime: lambda v: v.isoformat()},
        # Validate assignment
        validate_assignment=True,
        # Use enum values
        use_enum_values=True,
        # Allow extra fields (for future extensibility)
        extra='ignore'
    )

    identifier: str = Field(..., description="Normalized phone/email (E.164 for phones, lowercase for emails)", min_length=1)

    # Granular message counts
    individual_from_me: int = Field(0, description="1:1 messages I sent to this contact", ge=0)
    individual_to_me: int = Field(0, description="1:1 messages this contact sent to me", ge=0)
    group_from_me: int = Field(0, description="My messages in group chats with this contact", ge=0)
    group_to_me: int = Field(0, description="Their messages in group chats with me", ge=0)

    # Computed totals (for convenience)
    total_individual: int = Field(0, description="Total 1:1 messages (individual_from_me + individual_to_me)", ge=0)
    total_group: int = Field(0, description="Total group messages (group_from_me + group_to_me)", ge=0)
    total_messages: int = Field(0, description="Total message count (all categories)", ge=0)

    # Group chat details
    group_chat_names: List[str] = Field(default_factory=list, description="List of group chats user and contact share")

    # Last message info
    last_message_timestamp: Optional[datetime] = Field(None, description="When the most recent message was sent")
    last_message_text: Optional[str] = Field(None, description="Preview of the most recent message (truncated to 200 chars)")
    is_last_from_me: Optional[bool] = Field(None, description="Whether the most recent message was from the user")

    # Conversation state (V2 feature, always False for now)
    is_awaiting_response: bool = Field(False, description="Whether conversation is incomplete and user hasn't responded")

    def to_db_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database insertion/update"""
        return {
            "identifier": self.identifier,
            "individual_from_me": self.individual_from_me,
            "individual_to_me": self.individual_to_me,
            "group_from_me": self.group_from_me,
            "group_to_me": self.group_to_me,
            "total_individual": self.total_individual,
            "total_group": self.total_group,
            "total_messages": self.total_messages,
            "group_chat_names": self.group_chat_names,
            "last_message_timestamp": self.last_message_timestamp.isoformat() if self.last_message_timestamp else None,
            "last_message_text": self.last_message_text,
            "is_last_from_me": self.is_last_from_me,
            "is_awaiting_response": self.is_awaiting_response,
            "updated_at": datetime.now(timezone.utc).isoformat(),  # Always set to current time on update
        }

    @classmethod
    def from_db_dict(cls, data: Dict[str, Any]) -> "MessageStatsRecord":
        """Create MessageStatsRecord instance from database dictionary"""
        # Parse datetime fields if they're strings
        if isinstance(data.get("last_message_timestamp"), str):
            data["last_message_timestamp"] = datetime.fromisoformat(data["last_message_timestamp"])

        return cls(**data)


class ParticipantActivityStatsRecord(BaseModel):
    """
    Detailed activity statistics for a single participant within a group chat.

    Tracks messaging patterns including hourly distribution across a week,
    message length statistics, and other behavioral metrics.
    """
    model_config = ConfigDict(extra='ignore')

    message_count: int = Field(0, description="Total messages sent by this participant in this group", ge=0)
    avg_message_length: float = Field(0.0, description="Average character length of messages", ge=0)

    # Time-based distribution (0-indexed)
    hourly_distribution_utc: List[int] = Field(
        default_factory=lambda: [0] * 168,
        description="Message count by hour of week in UTC (0-167: Mon 00:00 to Sun 23:00)"
    )

    # Message timing
    first_message_timestamp: Optional[datetime] = Field(None, description="When this participant first messaged in this group")
    last_message_timestamp: Optional[datetime] = Field(None, description="When this participant last messaged in this group")


class GroupChatStatsRecord(BaseModel):
    """
    Aggregated statistics for a specific group chat.

    Group chats are uniquely identified by their participant set (not name or chat_identifier).
    This handles renamed chats and duplicate chats with the same participants.

    These statistics are computed and upserted during message logging.
    """
    model_config = ConfigDict(
        # Allow datetime objects to be serialized
        json_encoders={datetime: lambda v: v.isoformat()},
        # Validate assignment
        validate_assignment=True,
        # Use enum values
        use_enum_values=True,
        # Allow extra fields (for future extensibility)
        extra='ignore'
    )

    # Participant-based identifier (deterministic, handles renames/duplicates)
    group_chat_id: str = Field(
        ...,
        description="Derived from sorted participant list (e.g., '+15551111111|+15552222222|+15553333333')",
        min_length=1
    )
    participant_identifiers: List[str] = Field(
        default_factory=list,
        description="Sorted list of normalized participant identifiers (E.164 for phones, lowercase for emails)"
    )

    # Group chat metadata (can change over time)
    group_chat_name: str = Field(..., description="Most recent user-visible group chat name", min_length=1)
    chat_identifiers_seen: List[str] = Field(
        default_factory=list,
        description="All iMessage chat_identifiers observed for this participant set (handles duplicates)"
    )

    # Message counts
    messages_from_me: int = Field(0, description="Messages user sent in this group chat", ge=0)
    messages_to_me: int = Field(0, description="Messages user received in this group chat", ge=0)
    total_messages: int = Field(0, description="Total messages in this group chat (all participants)", ge=0)

    # Last message info
    last_message_timestamp: Optional[datetime] = Field(None, description="When the most recent message was sent")
    last_message_text: Optional[str] = Field(None, description="Preview of the most recent message (truncated to 200 chars)")
    last_message_sender: Optional[str] = Field(None, description="Identifier of who sent the last message")
    is_last_from_me: Optional[bool] = Field(None, description="Whether the most recent message was from the user")

    # Per-participant activity tracking
    participant_stats: Dict[str, ParticipantActivityStatsRecord] = Field(
        default_factory=dict,
        description="Detailed activity statistics per participant (identifier -> stats)"
    )

    @staticmethod
    def derive_group_chat_id(participant_identifiers: List[str]) -> str:
        """
        Derive a deterministic group_chat_id from participant list.

        Uses sorted participant identifiers joined with '|' for human-readability
        and debuggability. This ensures:
        - Same participants = same group_chat_id (regardless of name/chat_identifier)
        - Deterministic (alphabetically sorted)
        - Human-readable (can see who's in the chat)

        Args:
            participant_identifiers: List of normalized participant identifiers

        Returns:
            Deterministic group_chat_id string
        """
        sorted_participants = sorted(participant_identifiers)
        return "|".join(sorted_participants)

    def to_db_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database insertion/update"""
        # Serialize participant_stats to JSONB-compatible format
        participant_stats_serialized = {}
        for identifier, stats in self.participant_stats.items():
            participant_stats_serialized[identifier] = {
                "message_count": stats.message_count,
                "avg_message_length": stats.avg_message_length,
                "hourly_distribution_utc": stats.hourly_distribution_utc,
                "first_message_timestamp": stats.first_message_timestamp.isoformat() if stats.first_message_timestamp else None,
                "last_message_timestamp": stats.last_message_timestamp.isoformat() if stats.last_message_timestamp else None,
            }

        return {
            "group_chat_id": self.group_chat_id,
            "participant_identifiers": self.participant_identifiers,
            "group_chat_name": self.group_chat_name,
            "chat_identifiers_seen": self.chat_identifiers_seen,
            "messages_from_me": self.messages_from_me,
            "messages_to_me": self.messages_to_me,
            "total_messages": self.total_messages,
            "last_message_timestamp": self.last_message_timestamp.isoformat() if self.last_message_timestamp else None,
            "last_message_text": self.last_message_text,
            "last_message_sender": self.last_message_sender,
            "is_last_from_me": self.is_last_from_me,
            "participant_stats": participant_stats_serialized,
            "updated_at": datetime.now(timezone.utc).isoformat(),  # Always set to current time on update
        }

    @classmethod
    def from_db_dict(cls, data: Dict[str, Any]) -> "GroupChatStatsRecord":
        """Create GroupChatStatsRecord instance from database dictionary"""
        # Parse datetime fields if they're strings
        if isinstance(data.get("last_message_timestamp"), str):
            data["last_message_timestamp"] = datetime.fromisoformat(data["last_message_timestamp"])

        # Parse participant_stats from JSONB
        if "participant_stats" in data and isinstance(data["participant_stats"], dict):
            participant_stats = {}
            for identifier, stats_dict in data["participant_stats"].items():
                # Parse nested datetime fields
                if isinstance(stats_dict.get("first_message_timestamp"), str):
                    stats_dict["first_message_timestamp"] = datetime.fromisoformat(stats_dict["first_message_timestamp"])
                if isinstance(stats_dict.get("last_message_timestamp"), str):
                    stats_dict["last_message_timestamp"] = datetime.fromisoformat(stats_dict["last_message_timestamp"])

                participant_stats[identifier] = ParticipantActivityStatsRecord(**stats_dict)

            data["participant_stats"] = participant_stats

        return cls(**data)
