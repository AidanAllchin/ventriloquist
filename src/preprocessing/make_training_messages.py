"""
Create training messages from contacts and message database.

File: preprocessing/make_training_messages.py
Author: Aidan Allchin
Created: 2025-12-23
Last Modified: 2025-12-24
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Set

from dotenv import load_dotenv

from ..database import (
    detect_user_identifiers_from_db,
    get_group_chats_by_participants,
    get_message_texts_by_guids,
    get_messages_by_chat_identifiers,
    get_messages_with_identifiers,
)
from ..models import MessageRecord, TrainingMessage
from .utils import create_identifier_to_contact_map, load_contacts

log = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
MY_NAME = os.getenv("MY_NAME")
if not MY_NAME:
    raise ValueError("MY_NAME not found in .env")


async def detect_user_identifiers(contact_identifiers: Set[str]) -> Set[str]:
    """
    Detect user identifiers from environment or database.

    Tries in order:
    1. USER_IDENTIFIERS from .env (comma-separated list)
    2. Database detection via recipient_id from received messages

    Args:
        contact_identifiers: Set of all contact identifiers (to exclude from detection)

    Returns:
        Set of user identifiers
    """
    user_identifiers = set()

    # First, check if user specified their identifiers in .env
    env_identifiers = os.getenv("USER_IDENTIFIERS")
    if env_identifiers:
        user_identifiers = {id.strip() for id in env_identifiers.split(",") if id.strip()}
        log.info(f"Loaded {len(user_identifiers)} user identifiers from .env: {user_identifiers}")
        return user_identifiers

    # Fallback: detect from database
    log.info("USER_IDENTIFIERS not found in .env, detecting from database...")
    user_identifiers = await detect_user_identifiers_from_db(exclude_identifiers=contact_identifiers)

    if not user_identifiers:
        log.warning("Could not detect user identifiers! Consider adding USER_IDENTIFIERS to .env")
        log.warning("Example: USER_IDENTIFIERS=+1234567890,user@email.com")
    else:
        log.info(f"Detected {len(user_identifiers)} user identifiers from database")

    return user_identifiers


def format_reply_context(original_text: str, max_len: int = 50) -> Optional[str]:
    """
    Format reply context text, truncating if necessary.

    Args:
        original_text: The text of the message being replied to
        max_len: Maximum length before truncation

    Returns:
        Truncated text or None if no text
    """
    if not original_text:
        return None
    truncated = original_text[:max_len]
    if len(original_text) > max_len:
        truncated += "..."
    return truncated


def convert_message_to_training(
    msg: MessageRecord,
    identifier_map: Dict[str, str],
    user_identifiers: Set[str],
    is_group: bool = False,
    group_participants: Optional[List[str]] = None,
    guid_to_text: Optional[Dict[str, str]] = None,
) -> List[TrainingMessage]:
    """
    Convert a MessageRecord to TrainingMessage(s).

    Splits messages containing newlines into separate TrainingMessages,
    as iMessage often stores rapid-fire messages with \\n as a single message.

    Args:
        msg: MessageRecord to convert
        identifier_map: Mapping from identifier to contact name
        user_identifiers: Set of user's identifiers
        is_group: Whether this is a group chat message
        group_participants: List of participant identifiers for group chats
        guid_to_text: Mapping from GUID to message text for reply lookups

    Returns:
        List of TrainingMessage objects (usually 1, but more if content has newlines)
    """
    # Determine sender
    if msg.is_from_me:
        from_contact = MY_NAME
    else:
        # For received messages, sender_id contains the sender
        sender_id = msg.sender_id or "Unknown"
        from_contact = identifier_map.get(sender_id, sender_id)

    # Determine chat members (always include user's name for consistency)
    if is_group and group_participants:
        # Convert participant identifiers to names
        chat_members = set()
        for participant in group_participants:
            if participant not in user_identifiers:
                chat_members.add(identifier_map.get(participant, participant))
        chat_members.add(MY_NAME)  # Always include user's name
        chat_members = sorted(chat_members)
    else:
        # Individual chat - just the two participants
        if msg.is_from_me:
            other_person = msg.recipient_id or "Unknown"
            other_name = identifier_map.get(other_person, other_person)
        else:
            other_person = msg.sender_id or "Unknown"
            other_name = identifier_map.get(other_person, other_person)
        chat_members = sorted([MY_NAME, other_name])  # type: ignore

    # Resolve reply context (use thread_originator_guid for actual inline replies)
    reply_to_text = None
    if msg.thread_originator_guid and guid_to_text:
        original_text = guid_to_text.get(msg.thread_originator_guid)
        if original_text:
            reply_to_text = format_reply_context(original_text)

    # Keep full content - escaping is handled by json.dumps when rendering
    content = msg.text or ""
    chat_id = msg.chat_identifier or "unknown"
    timestamp = msg.timestamp.isoformat()

    return [
        TrainingMessage(
            chat_id=chat_id,
            from_contact=from_contact,  # type: ignore
            timestamp=timestamp,
            content=content,
            is_group_chat=is_group,
            chat_members=chat_members,
            reply_to_text=reply_to_text,
            thread_originator_guid=msg.thread_originator_guid,
        )
    ]


async def collect_individual_training_messages(
    contact_identifiers: List[str],
    identifier_map: Dict[str, str],
    user_identifiers: Set[str],
) -> List[TrainingMessage]:
    """
    Collect all individual (1:1) training messages with contacts.

    Args:
        contact_identifiers: List of contact identifiers
        identifier_map: Mapping from identifier to contact name
        user_identifiers: Set of user's identifiers

    Returns:
        List of TrainingMessage objects
    """
    log.info(f"Fetching individual messages for {len(contact_identifiers)} contacts...")
    messages = await get_messages_with_identifiers(contact_identifiers)

    # Collect all thread_originator_guids for batch lookup (actual inline replies)
    reply_guids = [msg.thread_originator_guid for msg in messages if msg.thread_originator_guid]
    guid_to_text = await get_message_texts_by_guids(reply_guids) if reply_guids else {}
    log.info(f"Resolved {len(guid_to_text)} inline reply references")

    training_messages = []
    for msg in messages:
        # Skip messages without text
        if not msg.text:
            continue

        training_msgs = convert_message_to_training(
            msg,
            identifier_map,
            user_identifiers,
            is_group=False,
            guid_to_text=guid_to_text,
        )
        training_messages.extend(training_msgs)

    log.info(f"Collected {len(training_messages)} individual training messages")
    return training_messages


async def collect_group_training_messages(
    contact_identifiers: Set[str],
    identifier_map: Dict[str, str],
    user_identifiers: Set[str],
) -> List[TrainingMessage]:
    """
    Collect all group chat training messages where every participant is a contact.

    Args:
        contact_identifiers: Set of contact identifiers
        identifier_map: Mapping from identifier to contact name
        user_identifiers: Set of user's identifiers

    Returns:
        List of TrainingMessage objects
    """
    log.info("Finding qualifying group chats...")

    # Get group chats where all non-user participants are contacts
    qualifying_groups = await get_group_chats_by_participants(
        required_identifiers=contact_identifiers,
        exclude_identifiers=user_identifiers,
    )

    if not qualifying_groups:
        log.info("No qualifying group chats found")
        return []

    log.info(f"Found {len(qualifying_groups)} qualifying group chats")

    # Collect all chat identifiers from qualifying groups
    all_chat_identifiers = []
    group_participants_map = {}  # Maps chat_identifier -> participants

    for group in qualifying_groups:
        for chat_id in group.chat_identifiers_seen:
            all_chat_identifiers.append(chat_id)
            group_participants_map[chat_id] = group.participant_identifiers

    # Fetch all messages from these chats
    log.info(f"Fetching messages from {len(all_chat_identifiers)} chat identifiers...")
    messages = await get_messages_by_chat_identifiers(all_chat_identifiers)

    # Collect all thread_originator_guids for batch lookup (actual inline replies)
    reply_guids = [msg.thread_originator_guid for msg in messages if msg.thread_originator_guid]
    guid_to_text = await get_message_texts_by_guids(reply_guids) if reply_guids else {}
    log.info(f"Resolved {len(guid_to_text)} inline reply references")

    training_messages = []
    for msg in messages:
        # Skip messages without text
        if not msg.text:
            continue

        # Get participants for this chat
        participants = group_participants_map.get(msg.chat_identifier, [])

        training_msgs = convert_message_to_training(
            msg,
            identifier_map,
            user_identifiers,
            is_group=True,
            group_participants=participants,
            guid_to_text=guid_to_text,
        )
        training_messages.extend(training_msgs)

    log.info(f"Collected {len(training_messages)} group chat training messages")
    return training_messages


async def make_training_messages(
    contacts_file: Path = Path("data/contacts_to_ids.jsonl"),
) -> List[TrainingMessage]:
    """
    Create all training messages from contacts.

    This includes:
    - All individual messages between the user and contacts
    - All group chat messages where every participant is a contact

    Args:
        contacts_file: Path to contacts JSONL file

    Returns:
        List of TrainingMessage objects sorted by timestamp
    """
    # Load contacts
    contacts = await load_contacts(contacts_file)
    identifier_map = create_identifier_to_contact_map(contacts)
    contact_identifiers = set(identifier_map.keys())

    log.info(f"Loaded {len(contacts)} contacts with {len(contact_identifiers)} unique identifiers")

    # Detect user identifiers
    user_identifiers = await detect_user_identifiers(contact_identifiers)

    # Collect individual messages
    individual_messages = await collect_individual_training_messages(
        list(contact_identifiers),
        identifier_map,
        user_identifiers,
    )

    # Collect group messages
    group_messages = await collect_group_training_messages(
        contact_identifiers,
        identifier_map,
        user_identifiers,
    )

    # Combine and sort by timestamp
    all_messages = individual_messages + group_messages
    all_messages.sort(key=lambda m: m.timestamp)

    log.info(f"\nTraining data collection complete:")
    log.info(f"  - {len(individual_messages)} individual messages")
    log.info(f"  - {len(group_messages)} group chat messages")
    log.info(f"  - {len(all_messages)} total messages")

    return all_messages
