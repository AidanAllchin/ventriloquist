"""
Shared data models for the Ventriloquist project.
"""

from .attachment import AttachmentForProcessing, AttachmentInfo, CachedDescription
from .contact import Contact
from .group_chat import GroupChatStatsRecord, ParticipantActivityStatsRecord
from .message import MessageRecord
from .message_stats import MessageStatsRecord
from .training_message import TrainingMessage, compute_delta_bucket

__all__ = [
    "AttachmentForProcessing",
    "AttachmentInfo",
    "CachedDescription",
    "Contact",
    "GroupChatStatsRecord",
    "MessageRecord",
    "MessageStatsRecord",
    "ParticipantActivityStatsRecord",
    "TrainingMessage",
    "compute_delta_bucket",
]
