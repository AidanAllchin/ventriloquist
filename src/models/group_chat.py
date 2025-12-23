"""
Group chat statistics models.

File: models/group_chat.py
Author: Aidan Allchin
Created: 2025-11-23
Last Modified: 2025-12-23
"""

from datetime import datetime, timezone
from typing import List, Dict, Optional, Any

from pydantic import BaseModel, Field, ConfigDict


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
        json_encoders={datetime: lambda v: v.isoformat()},
        validate_assignment=True,
        use_enum_values=True,
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
        """
        sorted_participants = sorted(participant_identifiers)
        return "|".join(sorted_participants)

    def to_db_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database insertion/update"""
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
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

    @classmethod
    def from_db_dict(cls, data: Dict[str, Any]) -> "GroupChatStatsRecord":
        """Create GroupChatStatsRecord instance from database dictionary"""
        if isinstance(data.get("last_message_timestamp"), str):
            data["last_message_timestamp"] = datetime.fromisoformat(data["last_message_timestamp"])

        if "participant_stats" in data and isinstance(data["participant_stats"], dict):
            participant_stats = {}
            for identifier, stats_dict in data["participant_stats"].items():
                if isinstance(stats_dict.get("first_message_timestamp"), str):
                    stats_dict["first_message_timestamp"] = datetime.fromisoformat(stats_dict["first_message_timestamp"])
                if isinstance(stats_dict.get("last_message_timestamp"), str):
                    stats_dict["last_message_timestamp"] = datetime.fromisoformat(stats_dict["last_message_timestamp"])

                participant_stats[identifier] = ParticipantActivityStatsRecord(**stats_dict)

            data["participant_stats"] = participant_stats

        return cls(**data)
