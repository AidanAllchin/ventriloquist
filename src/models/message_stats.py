"""
Message statistics record model.

File: models/message_stats.py
Author: Aidan Allchin
Created: 2025-11-23
Last Modified: 2025-12-23
"""

from datetime import datetime, timezone
from typing import List, Dict, Optional, Any

from pydantic import BaseModel, Field, ConfigDict


class MessageStatsRecord(BaseModel):
    """
    Aggregated message statistics for a specific identifier (phone/email).

    These statistics are computed and upserted after each batch of messages
    during the iMessage logging process.
    """
    model_config = ConfigDict(
        json_encoders={datetime: lambda v: v.isoformat()},
        validate_assignment=True,
        use_enum_values=True,
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
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

    @classmethod
    def from_db_dict(cls, data: Dict[str, Any]) -> "MessageStatsRecord":
        """Create MessageStatsRecord instance from database dictionary"""
        if isinstance(data.get("last_message_timestamp"), str):
            data["last_message_timestamp"] = datetime.fromisoformat(data["last_message_timestamp"])

        return cls(**data)
