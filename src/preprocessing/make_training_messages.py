"""
Create training messages from contacts and message database.

File: preprocessing/make_training_messages.py
Author: Aidan Allchin
Created: 2025-12-23
Last Modified: 2025-12-27
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Set

from dotenv import load_dotenv

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ..database import (
    detect_user_identifiers_from_db,
    get_cached_descriptions,
    get_group_chats_by_participants,
    get_messages_by_chat_identifiers,
    get_messages_with_identifiers,
)
from ..gemini.content_types import get_content_type
from ..models import MessageRecord, TrainingMessage, CachedDescription
from .utils import create_identifier_to_contact_map, load_contacts

console = Console()

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


# Reaction type codes from iMessage
# Object replacement character - iMessage uses this as placeholder for inline attachments
OBJECT_REPLACEMENT_CHAR = "\ufffc"

# Pattern for iMessage plugin payload attachments (rich link previews, app content)
# These are internal metadata with no semantic value - just UUIDs
PLUGIN_ATTACHMENT_SUFFIX = ".pluginPayloadAttachment"

REACTION_TYPES = {
    2000: "Loved",
    2001: "Liked",
    2002: "Disliked",
    2003: "Laughed",
    2004: "Emphasized",
    2005: "Questioned",
}


def is_reaction(msg: "MessageRecord") -> bool:
    """Check if a message is a reaction (tapback)."""
    return (
        msg.associated_message_type is not None
        and 2000 <= msg.associated_message_type < 3000
    )


def get_reaction_text(reaction_type: int) -> str:
    """Convert reaction type code to text."""
    return REACTION_TYPES.get(reaction_type, f"Reacted ({reaction_type})")


def convert_message_to_training(
    msg: MessageRecord,
    identifier_map: Dict[str, str],
    user_identifiers: Set[str],
    is_group: bool = False,
    group_participants: Optional[List[str]] = None,
    guid_to_content: Optional[Dict[str, str]] = None,
    attachment_cache: Optional[Dict[str, CachedDescription]] = None,
) -> List[TrainingMessage]:
    """
    Convert a MessageRecord to TrainingMessage(s).

    Handles three cases:
    1. Reactions: content_type="reaction", reply_to=original message content
    2. Regular messages with attachments: attachments first, then text
    3. Regular text messages

    Args:
        msg: MessageRecord to convert
        identifier_map: Mapping from identifier to contact name
        user_identifiers: Set of user's identifiers
        is_group: Whether this is a group chat message
        group_participants: List of participant identifiers for group chats
        guid_to_content: Mapping from message GUID to content (text or attachment description)
        attachment_cache: Mapping from attachment GUID to cached description

    Returns:
        List of TrainingMessage objects
    """
    # Determine sender
    if msg.is_from_me:
        from_contact = MY_NAME
    else:
        sender_id = msg.sender_id or "Unknown"
        from_contact = identifier_map.get(sender_id, sender_id)

    # Determine chat members
    if is_group and group_participants:
        chat_members = set()
        for participant in group_participants:
            if participant not in user_identifiers:
                chat_members.add(identifier_map.get(participant, participant))
        chat_members.add(MY_NAME)
        chat_members = sorted(chat_members)
    else:
        if msg.is_from_me:
            other_person = msg.recipient_id or "Unknown"
            other_name = identifier_map.get(other_person, other_person)
        else:
            other_person = msg.sender_id or "Unknown"
            other_name = identifier_map.get(other_person, other_person)
        chat_members = sorted([MY_NAME, other_name])  # type: ignore

    chat_id = msg.chat_identifier or "unknown"
    timestamp = msg.timestamp.isoformat()

    # Handle reactions
    if is_reaction(msg):
        # Resolve what we're reacting to
        reply_to = None
        if msg.associated_message_guid and guid_to_content:
            original_content = guid_to_content.get(msg.associated_message_guid)
            if original_content:
                reply_to = format_reply_context(original_content)
            else:
                # Can't find original message - skip this reaction
                return []
        else:
            # No GUID to look up - skip
            return []

        return [
            TrainingMessage(
                chat_id=chat_id,
                from_contact=from_contact,  # type: ignore
                timestamp=timestamp,
                content=get_reaction_text(msg.associated_message_type),  # type: ignore
                content_type="reaction",
                is_group_chat=is_group,
                chat_members=chat_members,
                reply_to_text=reply_to,
                thread_originator_guid=msg.associated_message_guid,
            )
        ]

    # Resolve reply context for regular messages (replies and inline replies)
    reply_to = None
    reply_guid = msg.thread_originator_guid
    if reply_guid and guid_to_content:
        original_content = guid_to_content.get(reply_guid)
        if original_content:
            reply_to = format_reply_context(original_content)

    results = []

    # Process attachments first (if any)
    if msg.attachments and attachment_cache:
        first_attachment = True
        for attachment in msg.attachments:
            cached = attachment_cache.get(attachment.guid)
            if cached and cached.description:
                content_type = cached.content_type
                content = cached.description
            else:
                content_type = get_content_type(attachment.mime_type, attachment.uti)
                if content_type == "audio" and not msg.is_audio_message and attachment.transfer_name:
                    content = attachment.transfer_name
                else:
                    content = "[NA]"

            # Skip plugin payload attachments (rich link previews) - they're just UUIDs
            if content.endswith(PLUGIN_ATTACHMENT_SUFFIX):
                continue

            # First attachment carries reply context if this message is a reply
            results.append(
                TrainingMessage(
                    chat_id=chat_id,
                    from_contact=from_contact,  # type: ignore
                    timestamp=timestamp,
                    content=content,
                    content_type=content_type,
                    is_group_chat=is_group,
                    chat_members=chat_members,
                    reply_to_text=reply_to if first_attachment else None,
                    thread_originator_guid=reply_guid if first_attachment else None,
                )
            )
            first_attachment = False

    # Add text message if present (filter out object replacement characters)
    if msg.text:
        # Remove object replacement characters (used as inline attachment placeholders)
        clean_text = msg.text.replace(OBJECT_REPLACEMENT_CHAR, "").strip()
        if clean_text:
            # If we already added attachments with reply context, text doesn't repeat it
            text_reply_to = reply_to if not results else None
            results.append(
                TrainingMessage(
                    chat_id=chat_id,
                    from_contact=from_contact,  # type: ignore
                    timestamp=timestamp,
                    content=clean_text,
                    content_type="text",
                    is_group_chat=is_group,
                    chat_members=chat_members,
                    reply_to_text=text_reply_to,
                    thread_originator_guid=reply_guid if text_reply_to else None,
                )
            )

    return results


def build_guid_to_content(
    messages: List[MessageRecord],
    attachment_cache: Dict[str, CachedDescription],
) -> Dict[str, str]:
    """
    Build a mapping from message GUID to content string.

    For text messages: uses the text
    For attachment-only messages: uses the first attachment's description

    Args:
        messages: List of messages to build mapping from
        attachment_cache: Cached attachment descriptions

    Returns:
        Dict mapping message GUID to content string
    """
    guid_to_content = {}
    for msg in messages:
        content = None

        # Clean text of object replacement characters
        clean_text = None
        if msg.text:
            clean_text = msg.text.replace(OBJECT_REPLACEMENT_CHAR, "").strip()

        # Prefer clean text content
        if clean_text:
            content = clean_text
        # Fall back to first attachment description (skip plugin attachments)
        elif msg.attachments:
            for att in msg.attachments:
                cached = attachment_cache.get(att.guid)
                if cached and cached.description:
                    # Skip plugin payload attachments
                    if cached.description.endswith(PLUGIN_ATTACHMENT_SUFFIX):
                        continue
                    content = cached.description
                    break

        if content:
            guid_to_content[msg.guid] = content

    return guid_to_content


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

    # Collect all attachment GUIDs for batch lookup
    attachment_guids = []
    for msg in messages:
        if msg.attachments:
            attachment_guids.extend(a.guid for a in msg.attachments)
    attachment_cache = await get_cached_descriptions(attachment_guids) if attachment_guids else {}
    log.info(f"Loaded {len(attachment_cache)} cached attachment descriptions")

    # Build guid_to_content mapping for reply/reaction lookups
    guid_to_content = build_guid_to_content(messages, attachment_cache)
    log.info(f"Built content mapping for {len(guid_to_content)} messages")

    training_messages = []
    for msg in messages:
        # Skip messages without text AND without attachments AND not a reaction
        if not msg.text and not msg.attachments and not is_reaction(msg):
            continue

        training_msgs = convert_message_to_training(
            msg,
            identifier_map,
            user_identifiers,
            is_group=False,
            guid_to_content=guid_to_content,
            attachment_cache=attachment_cache,
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

    # Collect all attachment GUIDs for batch lookup
    attachment_guids = []
    for msg in messages:
        if msg.attachments:
            attachment_guids.extend(a.guid for a in msg.attachments)
    attachment_cache = await get_cached_descriptions(attachment_guids) if attachment_guids else {}
    log.info(f"Loaded {len(attachment_cache)} cached attachment descriptions")

    # Build guid_to_content mapping for reply/reaction lookups
    guid_to_content = build_guid_to_content(messages, attachment_cache)
    log.info(f"Built content mapping for {len(guid_to_content)} messages")

    training_messages = []
    for msg in messages:
        # Skip messages without text AND without attachments AND not a reaction
        if not msg.text and not msg.attachments and not is_reaction(msg):
            continue

        # Get participants for this chat
        participants = group_participants_map.get(msg.chat_identifier, [])

        training_msgs = convert_message_to_training(
            msg,
            identifier_map,
            user_identifiers,
            is_group=True,
            group_participants=participants,
            guid_to_content=guid_to_content,
            attachment_cache=attachment_cache,
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

    # Rich summary output
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Label", style="dim")
    table.add_column("Count", style="bold cyan", justify="right")
    table.add_row("Individual messages", f"{len(individual_messages):,}")
    table.add_row("Group chat messages", f"{len(group_messages):,}")
    table.add_row("Total messages", f"{len(all_messages):,}")

    console.print(Panel(table, title="[bold green]Training Data Collection Complete", border_style="green"))

    return all_messages
