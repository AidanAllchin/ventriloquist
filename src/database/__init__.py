"""
File: database/__init__.py
Author: Aidan Allchin
Created: 2025-12-23
Last Modified: 2025-12-23
"""

from .create_tables import init_local_database
from .group_chat_stats import (
    fetch_existing_group_chat_statistics,
    upsert_group_chat_statistics,
    get_top_group_chats,
    get_group_chats_by_participants,
)
from .messages import (
    get_last_synced_timestamp,
    sync_batch_to_local_db,
    get_messages_with_identifiers,
    get_messages_by_chat_identifiers,
    get_message_texts_by_guids,
    detect_user_identifiers_from_db,
)
from .message_stats import (
    fetch_existing_statistics,
    upsert_statistics,
    get_top_contacts,
)
from .training_data import (
    store_training_messages,
    get_training_messages_by_chat,
    get_all_chat_ids,
    get_chat_metadata,
)
from .attachment_cache import (
    get_cached_description,
    get_cached_descriptions,
    cache_description,
    cache_descriptions_batch,
    get_processing_stats,
    clear_errors,
)

__all__ = [
    "init_local_database",
    "fetch_existing_group_chat_statistics",
    "upsert_group_chat_statistics",
    "get_top_group_chats",
    "get_group_chats_by_participants",
    "get_last_synced_timestamp",
    "sync_batch_to_local_db",
    "get_messages_with_identifiers",
    "get_messages_by_chat_identifiers",
    "get_message_texts_by_guids",
    "detect_user_identifiers_from_db",
    "fetch_existing_statistics",
    "upsert_statistics",
    "get_top_contacts",
    "store_training_messages",
    "get_training_messages_by_chat",
    "get_all_chat_ids",
    "get_chat_metadata",
    "get_cached_description",
    "get_cached_descriptions",
    "cache_description",
    "cache_descriptions_batch",
    "get_processing_stats",
    "clear_errors",
]
