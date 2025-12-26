"""
Inference module for Ventriloquist.

Provides multiple inference modes:
- Interactive: Chat with synthetic context
- Auto: Watch model generate conversations
- Live Chat: Chat with real iMessage context
"""

from .live_chat import live_chat
from .interactive import (
    interactive_mode,
    auto_mode,
    generate_response,
    build_prompt,
    GeneratedMessage,
)
from .utils import load_model, get_device
from .imessage import check_imessage_access, list_chats, fetch_recent_messages
from .formatting import messages_to_prompt, compute_delta
from .checkpoints import find_checkpoints, select_checkpoint, Checkpoint

__all__ = [
    # Modes
    "live_chat",
    "interactive_mode",
    "auto_mode",
    # Model loading
    "load_model",
    "get_device",
    "generate_response",
    "build_prompt",
    "GeneratedMessage",
    # iMessage
    "check_imessage_access",
    "list_chats",
    "fetch_recent_messages",
    # Formatting
    "messages_to_prompt",
    "compute_delta",
    # Checkpoints
    "find_checkpoints",
    "select_checkpoint",
    "Checkpoint",
]
