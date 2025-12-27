#!/usr/bin/env python3
"""
iMessage logger that logs messages to an SQLite database

File: collection/imessage_logger.py
Author: Aidan Allchin
Created: 2025-10-02
Last Modified: 2025-12-27
"""

import os
import sys
import asyncio
import aiosqlite
import logging
import fcntl
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv

from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
    MofNCompleteColumn
)

from ..database import (
    init_local_database,
    get_last_synced_timestamp,
    sync_batch_to_local_db,
    fetch_existing_group_chat_statistics,
    fetch_existing_statistics,
    upsert_statistics,
    upsert_group_chat_statistics,
)
from .id_normalization import normalize_identifier
from ..models import (
    MessageRecord,
    MessageStatsRecord,
    GroupChatStatsRecord,
    ParticipantActivityStatsRecord,
    AttachmentInfo,
)

# Load environment variables
load_dotenv()

# Configure logging
LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / f"simple_imessage_logger_{datetime.now().strftime('%Y-%m-%d')}.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

# Configuration
IMESSAGE_DB_PATH = os.path.expanduser("~/Library/Messages/chat.db")
LOCK_FILE = Path(__file__).parent / "logs" / "imessage_logger.lock"
BATCH_SIZE = 500  # Number of messages to insert per batch

# Rich console for pretty output
console = Console()


class iMessageLogger:
    def __init__(self):
        self.user_identifiers = []  # Will be populated with user's phone/email

    def _imessage_timestamp_to_datetime(self, timestamp: int) -> Optional[datetime]:
        """
        Convert iMessage timestamp to datetime object.

        IMPORTANT: iMessage stores timestamps as nanoseconds since 2001-01-01 00:00:00 UTC.
        The underlying timestamps are absolute UTC times, NOT local times.

        When you view messages in SQLite with 'localtime', it converts to your CURRENT timezone,
        not the timezone you were in when the message was sent. For example:
        - A message sent in Hawaii (UTC-10) at 8:23 PM HST
        - Is stored as 06:23:12 UTC (next day)
        - Will display as 11:23 PM PDT if you're currently in Pacific timezone

        This function correctly returns UTC datetimes, which is what we want for consistent
        storage in the database regardless of user timezone.
        """
        if timestamp == 0:
            return None
        # Convert from nanoseconds since 2001-01-01 to Unix timestamp (seconds since 1970-01-01)
        unix_timestamp = (timestamp / 1_000_000_000) + 978307200
        # Return as UTC datetime
        return datetime.fromtimestamp(unix_timestamp, tz=timezone.utc)

    def _datetime_to_imessage_timestamp(self, dt: datetime) -> int:
        """
        Convert datetime to iMessage timestamp format.

        iMessage timestamps are stored as nanoseconds since 2001-01-01 00:00:00 UTC.
        This converts a datetime to that format.
        """
        if dt.tzinfo is None or dt.tzinfo.utcoffset(dt) is None:
            dt = dt.replace(tzinfo=timezone.utc)
        # Convert to Unix timestamp, then adjust for 2001 epoch and convert to nanoseconds
        unix_timestamp = dt.timestamp() - 978307200
        return int(unix_timestamp * 1_000_000_000)

    def _process_text(self, text: Optional[str], attributed_body: Optional[bytes]) -> str:
        """Extract text from message, handling attributedBody if needed"""
        if text is not None and text.strip() != "":
            return text

        if attributed_body is None:
            return ""

        try:
            bytes_text = attributed_body.split(b"NSString")[1][5:]
            if bytes_text[0] == 129:
                length = int.from_bytes(bytes_text[1:3], "little")
                bytes_text = bytes_text[3 : length + 3]
            else:
                length = bytes_text[0]
                bytes_text = bytes_text[1 : length + 1]
            return bytes_text.decode()
        except Exception as e:
            # This is expected for messages with only attachments
            log.debug(f"Could not extract text from attributedBody (likely attachment-only message)")
            return ""

    async def _get_user_identifiers(self) -> List[str]:
        """
        Get the user's phone numbers/emails from the iMessage database.
        These are used as fallback when destination_caller_id is NULL.
        """
        if self.user_identifiers:
            return self.user_identifiers

        query = """
        SELECT DISTINCT destination_caller_id, COUNT(*) as count
        FROM message
        WHERE destination_caller_id IS NOT NULL
        AND is_from_me = 0
        GROUP BY destination_caller_id
        ORDER BY count DESC
        """

        try:
            async with aiosqlite.connect(IMESSAGE_DB_PATH) as conn:
                async with conn.execute(query) as cursor:
                    rows = await cursor.fetchall()
                    self.user_identifiers = [row[0] for row in rows if row[0]]
                    if self.user_identifiers:
                        log.debug(f"Found user identifiers: {self.user_identifiers}")
                    return self.user_identifiers
        except Exception as e:
            log.warning(f"Could not fetch user identifiers: {e}")
            return []

    async def _get_chat_participants(self, chat_identifier: str, conn: aiosqlite.Connection) -> List[str]:
        """
        Get all participant phone numbers/emails for a given chat.
        This is especially useful for group chats.

        Args:
            chat_identifier: The identifier of the chat to get participants for
            conn: The connection to the iMessage database

        Returns:
            A list of participant phone numbers/emails
        """
        query = """
        SELECT DISTINCT handle.id
        FROM chat_handle_join
        JOIN handle ON chat_handle_join.handle_id = handle.ROWID
        JOIN chat ON chat_handle_join.chat_id = chat.ROWID
        WHERE chat.chat_identifier = ?
        AND handle.id IS NOT NULL
        ORDER BY handle.id
        """

        try:
            async with conn.execute(query, (chat_identifier,)) as cursor:
                rows = await cursor.fetchall()
                return [row[0] for row in rows if row[0]]
        except Exception as e:
            log.warning(f"Could not fetch participants for chat {chat_identifier}: {e}")
            return []

    async def _fetch_new_messages(self, since_timestamp: Optional[datetime]) -> List[MessageRecord]:
        """Fetch new messages from the iMessage database since the given timestamp"""
        if since_timestamp is None:
            # No previous sync - get ALL messages (use epoch zero)
            since_timestamp = datetime(1970, 1, 1, tzinfo=timezone.utc)

        # Get user identifiers for fallback
        user_ids = await self._get_user_identifiers()
        primary_user_id = user_ids[0] if user_ids else "me"

        since_imessage_ts = self._datetime_to_imessage_timestamp(since_timestamp)

        query = """
        SELECT
            message.ROWID,
            message.guid,
            message.text,
            message.date,
            message.date_read,
            message.date_delivered,
            handle.id as sender_id,
            message.destination_caller_id,
            message.is_from_me,
            message.service,
            message.attributedBody,
            chat.chat_identifier,
            chat.display_name,
            chat.style,
            message.cache_has_attachments,
            message.is_read,
            message.reply_to_guid,
            message.thread_originator_guid,
            message.is_audio_message,
            GROUP_CONCAT(attachment.guid, '|||') as attachment_guids,
            GROUP_CONCAT(attachment.mime_type, '|||') as attachment_mimes,
            GROUP_CONCAT(attachment.filename, '|||') as attachment_paths,
            GROUP_CONCAT(attachment.uti, '|||') as attachment_utis,
            GROUP_CONCAT(attachment.transfer_name, '|||') as attachment_names,
            GROUP_CONCAT(attachment.is_sticker, '|||') as attachment_stickers,
            message.associated_message_type,
            REPLACE(REPLACE(message.associated_message_guid, 'p:0/', ''), 'bp:', '') as associated_message_guid
        FROM
            message
        LEFT JOIN
            handle ON message.handle_id = handle.ROWID
        LEFT JOIN
            chat ON message.cache_roomnames = chat.chat_identifier
        LEFT JOIN
            message_attachment_join ON message.ROWID = message_attachment_join.message_id
        LEFT JOIN
            attachment ON message_attachment_join.attachment_id = attachment.ROWID
        WHERE
            message.date > ?
        GROUP BY
            message.ROWID
        ORDER BY
            message.date ASC
        """

        messages = []
        corrupted_messages = 0
        try:
            async with aiosqlite.connect(IMESSAGE_DB_PATH) as conn:
                async with conn.execute(query, (since_imessage_ts,)) as cursor:
                    async for row in cursor:
                        # Parse the row
                        message_id = row[0]
                        guid = row[1]
                        raw_text = row[2]
                        attributed_body = row[10]

                        # Process text - handle NULL, empty string, and attributedBody
                        text = self._process_text(raw_text, attributed_body)

                        # Parse other fields first to determine attachment handling
                        has_attachments = bool(row[14])

                        timestamp = self._imessage_timestamp_to_datetime(row[3]) if row[3] else None
                        date_read = self._imessage_timestamp_to_datetime(row[4]) if row[4] else None
                        date_delivered = self._imessage_timestamp_to_datetime(row[5]) if row[5] else None
                        handle_id = row[6]  # Phone/email from handle table
                        destination_caller_id = row[7]  # Your number
                        is_from_me = bool(row[8])
                        service = row[9]

                        # Determine actual sender and recipient based on direction
                        # For SENT messages: handle.id = recipient, destination_caller_id = you
                        # For RECEIVED messages: handle.id = sender, destination_caller_id = you
                        if is_from_me:
                            sender_id = destination_caller_id or primary_user_id  # You (fallback to primary ID)
                            recipient_id = handle_id or "group"  # The person you sent to, or "group" for group messages
                        else:
                            sender_id = handle_id or "unknown"  # The person who sent it
                            recipient_id = destination_caller_id or primary_user_id  # You (fallback to primary ID)

                        # Normalize identifiers to E.164 (phones) or lowercase (emails)
                        if sender_id:
                            normalized_sender = normalize_identifier(sender_id)
                            if normalized_sender:
                                sender_id = normalized_sender

                        if recipient_id:
                            normalized_recipient = normalize_identifier(recipient_id)
                            if normalized_recipient:
                                recipient_id = normalized_recipient

                        raw_chat_identifier = row[11]
                        group_chat_display_name = row[12]
                        chat_style = row[13]
                        is_read = bool(row[15])
                        reply_to_guid = row[16]
                        thread_originator_guid = row[17]
                        is_audio_message = bool(row[18])
                        associated_message_type = row[25]  # Reaction type (2000=Loved, etc.)
                        associated_message_guid = row[26]  # GUID of message being reacted to

                        # Parse attachment data from GROUP_CONCAT fields
                        attachments = None
                        attachment_guids = row[19]
                        if attachment_guids:
                            attachment_mimes = row[20]
                            attachment_paths = row[21]
                            attachment_utis = row[22]
                            attachment_names = row[23]
                            attachment_stickers = row[24]

                            # Split by delimiter and zip together
                            guids = attachment_guids.split('|||')
                            mimes = (attachment_mimes or '').split('|||') if attachment_mimes else [''] * len(guids)
                            paths = (attachment_paths or '').split('|||') if attachment_paths else [''] * len(guids)
                            utis = (attachment_utis or '').split('|||') if attachment_utis else [''] * len(guids)
                            names = (attachment_names or '').split('|||') if attachment_names else [''] * len(guids)
                            stickers = (attachment_stickers or '').split('|||') if attachment_stickers else ['0'] * len(guids)

                            attachments = []
                            for i, att_guid in enumerate(guids):
                                if att_guid:  # Skip empty entries
                                    attachments.append(AttachmentInfo(
                                        guid=att_guid,
                                        mime_type=mimes[i] if i < len(mimes) and mimes[i] else None,
                                        uti=utis[i] if i < len(utis) and utis[i] else None,
                                        filename=paths[i] if i < len(paths) and paths[i] else None,
                                        transfer_name=names[i] if i < len(names) and names[i] else None,
                                        is_sticker=bool(int(stickers[i])) if i < len(stickers) and stickers[i] else False,
                                    ))

                        # Determine if group chat (style 43 = group, 45 = individual)
                        is_group_chat = chat_style == 43 if chat_style else False

                        # Set chat_identifier - for individual chats, use the other person's number
                        # For group chats, use the chat identifier from the DB
                        group_chat_participants = None
                        if is_group_chat:
                            chat_identifier = raw_chat_identifier
                            group_chat_name = group_chat_display_name or f"Unnamed Group Chat"

                            # Fetch all participants for this group chat
                            if chat_identifier:
                                raw_participants = await self._get_chat_participants(chat_identifier, conn)
                                # Normalize all participant identifiers
                                group_chat_participants = []
                                for participant in raw_participants:
                                    normalized = normalize_identifier(participant)
                                    if normalized:
                                        group_chat_participants.append(normalized)
                        else:
                            # For individual chats, use the other person's phone/email as identifier
                            # This is handle_id for both sent and received messages
                            chat_identifier = handle_id or raw_chat_identifier or "unknown"
                            group_chat_name = None

                        # Skip truly empty messages (no text AND no attachments)
                        # These are common in the iMessage DB and should be filtered out
                        if (not text or text.strip() == "") and not has_attachments:
                            log.debug(f"Skipping empty message: {guid}")
                            corrupted_messages += 1
                            continue

                        # Skip corrupted received messages with no sender
                        if sender_id is None and not is_from_me:
                            log.debug(f"Skipping corrupted message: {guid}")
                            corrupted_messages += 1
                            continue

                        if timestamp is None:
                            log.warning(f"Skipping message with no timestamp: {guid}")
                            continue

                        messages.append(MessageRecord(
                            message_id=message_id,
                            guid=guid,
                            text=text,
                            timestamp=timestamp,
                            sender_id=sender_id,
                            recipient_id=recipient_id,
                            is_from_me=is_from_me,
                            service=service,
                            chat_identifier=chat_identifier,
                            is_group_chat=is_group_chat,
                            group_chat_name=group_chat_name,
                            group_chat_participants=group_chat_participants,
                            has_attachments=has_attachments,
                            is_audio_message=is_audio_message,
                            attachments=attachments,
                            is_read=is_read,
                            date_read=date_read,
                            date_delivered=date_delivered,
                            reply_to_guid=reply_to_guid,
                            thread_originator_guid=thread_originator_guid,
                            associated_message_type=associated_message_type,
                            associated_message_guid=associated_message_guid,
                        ))
            log.info(f"Skipped {corrupted_messages} corrupted messages")
        except Exception as e:
            log.error(f"Error fetching messages from iMessage DB: {e}")
            raise

        return messages

    def _compute_statistics_from_messages(
            self,
            messages: List[MessageRecord],
        ) -> Dict[str, MessageStatsRecord]:
        """
        Compute message statistics from a batch of messages. This computes
        statistics for the BATCH only, not cumulative.

        Attribution logic:
        - Individual messages I send -> recipient (individual_from_me)
        - Individual messages I receive -> sender (individual_to_me)
        - Group messages I send -> EACH participant in group (group_from_me)
        - Group messages I receive -> sender (group_to_me)

        Args:
            messages: List of MessageRecord objects to compute statistics from

        Returns:
            Dictionary mapping identifier to MessageStatsRecord
        """
        # Dictionary to accumulate stats per identifier
        stats_dict: Dict[str, Dict[str, Any]] = {}

        for msg in messages:
            # Determine which identifiers to attribute this message to
            identifiers_to_update = []

            if msg.is_group_chat:
                if msg.is_from_me:
                    # Group message I sent: attribute to EACH participant in the group
                    if msg.group_chat_participants:
                        identifiers_to_update = [(id, "group_from_me") for id in msg.group_chat_participants]
                else:
                    # Group message I received: attribute to the sender
                    if msg.sender_id and msg.sender_id not in ["unknown", "group"]:
                        identifiers_to_update = [(msg.sender_id, "group_to_me")]
            else:
                # Individual chat
                if msg.is_from_me:
                    # Individual message I sent: attribute to recipient
                    if msg.recipient_id and msg.recipient_id not in ["unknown", "group"]:
                        identifiers_to_update = [(msg.recipient_id, "individual_from_me")]
                else:
                    # Individual message I received: attribute to sender
                    if msg.sender_id and msg.sender_id not in ["unknown", "group"]:
                        identifiers_to_update = [(msg.sender_id, "individual_to_me")]

            # Update stats for each identifier
            for identifier, stat_field in identifiers_to_update:
                # Initialize stats for this identifier if needed
                if identifier not in stats_dict:
                    stats_dict[identifier] = {
                        "identifier": identifier,
                        "individual_from_me": 0,
                        "individual_to_me": 0,
                        "group_from_me": 0,
                        "group_to_me": 0,
                        "total_individual": 0,
                        "total_group": 0,
                        "total_messages": 0,
                        "group_chat_names": set(),  # Use set to avoid duplicates
                        "last_message_timestamp": None,
                        "last_message_text": None,
                        "is_last_from_me": None,
                        "is_awaiting_response": False,
                    }

                stats = stats_dict[identifier]

                # Update the specific granular count
                stats[stat_field] += 1

                # Update computed totals
                if msg.is_group_chat:
                    stats["total_group"] += 1
                    # Add group chat name to set
                    if msg.group_chat_name:
                        stats["group_chat_names"].add(msg.group_chat_name)
                else:
                    stats["total_individual"] += 1

                stats["total_messages"] += 1

                # Update last message info if this is the most recent
                # For group messages I sent, we use the message timestamp for each participant
                # but we don't necessarily update "last_message_text" for group messages I sent
                # since that would be confusing (same message appearing for multiple contacts)
                # Instead, we only update last_message_text/is_last_from_me for 1:1 messages
                # or group messages sent TO me (not from me)
                should_update_last_message = (
                    stats["last_message_timestamp"] is None or
                    msg.timestamp > stats["last_message_timestamp"]
                )

                if should_update_last_message:
                    # Always update timestamp
                    stats["last_message_timestamp"] = msg.timestamp

                    # Only update text/direction for non-group-from-me messages
                    # This prevents group messages I sent from showing as "last message" for all participants
                    if stat_field != "group_from_me":
                        stats["last_message_text"] = msg.text[:200] if msg.text else None
                        stats["is_last_from_me"] = msg.is_from_me
                    elif stats["last_message_text"] is None:
                        # If we have no last message yet, use this one even if it's group_from_me
                        stats["last_message_text"] = msg.text[:200] if msg.text else None
                        stats["is_last_from_me"] = msg.is_from_me

        # Convert to MessageStatsRecord objects
        result = {}
        for identifier, stats in stats_dict.items():
            # Convert set to sorted list for group_chat_names
            stats["group_chat_names"] = sorted(list(stats["group_chat_names"]))

            result[identifier] = MessageStatsRecord(**stats)

        return result

    def _merge_statistics(
            self,
            existing: Optional[MessageStatsRecord],
            batch: MessageStatsRecord
        ) -> MessageStatsRecord:
        """
        Merge batch statistics with existing statistics.

        Args:
            existing: Existing statistics from database (None if first time)
            batch: New statistics computed from batch

        Returns:
            Merged MessageStatsRecord
        """
        if existing is None:
            return batch

        # Add counts together
        merged_data = {
            "identifier": batch.identifier,
            "individual_from_me": existing.individual_from_me + batch.individual_from_me,
            "individual_to_me": existing.individual_to_me + batch.individual_to_me,
            "group_from_me": existing.group_from_me + batch.group_from_me,
            "group_to_me": existing.group_to_me + batch.group_to_me,
            "total_individual": existing.total_individual + batch.total_individual,
            "total_group": existing.total_group + batch.total_group,
            "total_messages": existing.total_messages + batch.total_messages,
            # Merge group chat names (union of both sets)
            "group_chat_names": sorted(list(set(existing.group_chat_names) | set(batch.group_chat_names))),
            # Use the most recent last_message info
            "last_message_timestamp": batch.last_message_timestamp if (
                batch.last_message_timestamp and (
                    existing.last_message_timestamp is None or
                    batch.last_message_timestamp > existing.last_message_timestamp
                )
            ) else existing.last_message_timestamp,
            "last_message_text": batch.last_message_text if (
                batch.last_message_timestamp and (
                    existing.last_message_timestamp is None or
                    batch.last_message_timestamp > existing.last_message_timestamp
                )
            ) else existing.last_message_text,
            "is_last_from_me": batch.is_last_from_me if (
                batch.last_message_timestamp and (
                    existing.last_message_timestamp is None or
                    batch.last_message_timestamp > existing.last_message_timestamp
                )
            ) else existing.is_last_from_me,
            "is_awaiting_response": False,  # V2 feature, always False for now
        }

        return MessageStatsRecord(**merged_data)

    def _compute_group_chat_statistics_from_messages(
            self,
            messages: List[MessageRecord],
        ) -> Dict[str, GroupChatStatsRecord]:
        """
        Compute group chat statistics from a batch of messages. This computes
        statistics for the BATCH only, not cumulative.

        Only processes group chat messages. Creates one stat record per unique
        group chat (identified by participant set).

        Args:
            messages: List of MessageRecord objects to compute statistics from

        Returns:
            Dictionary mapping group_chat_id to GroupChatStatsRecord
        """
        # Dictionary to accumulate stats per group chat
        stats_dict: Dict[str, Dict[str, Any]] = {}

        for msg in messages:
            # Only process group chat messages
            if not msg.is_group_chat or not msg.group_chat_participants:
                continue

            # Derive group_chat_id from participants
            group_chat_id = GroupChatStatsRecord.derive_group_chat_id(msg.group_chat_participants)

            # Initialize stats for this group chat if needed
            if group_chat_id not in stats_dict:
                stats_dict[group_chat_id] = {
                    "group_chat_id": group_chat_id,
                    "participant_identifiers": sorted(msg.group_chat_participants),
                    "group_chat_name": msg.group_chat_name or "Unnamed Group Chat",
                    "chat_identifiers_seen": set(),  # Use set to avoid duplicates
                    "messages_from_me": 0,
                    "messages_to_me": 0,
                    "total_messages": 0,
                    "last_message_timestamp": None,
                    "last_message_text": None,
                    "last_message_sender": None,
                    "is_last_from_me": None,
                    "participant_stats": {},  # Will store ParticipantActivityStatsRecord objects
                }

            stats = stats_dict[group_chat_id]

            # Update group chat name to most recent (handles renames)
            if msg.group_chat_name:
                stats["group_chat_name"] = msg.group_chat_name

            # Track chat_identifier
            if msg.chat_identifier:
                stats["chat_identifiers_seen"].add(msg.chat_identifier)

            # Update message counts
            if msg.is_from_me:
                stats["messages_from_me"] += 1
            else:
                stats["messages_to_me"] += 1

            stats["total_messages"] += 1

            # Update last message info if this is the most recent
            should_update_last_message = (
                stats["last_message_timestamp"] is None or
                msg.timestamp > stats["last_message_timestamp"]
            )

            if should_update_last_message:
                stats["last_message_timestamp"] = msg.timestamp
                stats["last_message_text"] = msg.text[:200] if msg.text else None
                stats["last_message_sender"] = msg.sender_id
                stats["is_last_from_me"] = msg.is_from_me

            # Update per-participant statistics
            sender_id = msg.sender_id
            if not sender_id or sender_id in ["unknown", "group"]:
                continue  # Skip if no valid sender

            # Initialize participant stats if needed
            if sender_id not in stats["participant_stats"]:
                stats["participant_stats"][sender_id] = {
                    "message_count": 0,
                    "avg_message_length": 0.0,
                    "hourly_distribution_utc": [0] * 168,
                    "first_message_timestamp": None,
                    "last_message_timestamp": None,
                    "_total_message_length": 0,  # Temporary field for computing average
                }

            participant_stats = stats["participant_stats"][sender_id]

            # Update participant message count
            participant_stats["message_count"] += 1

            # Update message length average
            if msg.text:
                text_length = len(msg.text)
                participant_stats["_total_message_length"] += text_length
                participant_stats["avg_message_length"] = (
                    participant_stats["_total_message_length"] / participant_stats["message_count"]
                )

            # Update hourly distribution (168-hour week in UTC)
            # Index calculation: day_of_week * 24 + hour
            weekday = msg.timestamp.weekday()  # 0 = Monday, 6 = Sunday
            hour = msg.timestamp.hour
            hour_index = weekday * 24 + hour
            participant_stats["hourly_distribution_utc"][hour_index] += 1

            # Update first/last message timestamps for this participant
            if participant_stats["first_message_timestamp"] is None:
                participant_stats["first_message_timestamp"] = msg.timestamp
            participant_stats["last_message_timestamp"] = msg.timestamp

        # Convert to GroupChatStatsRecord objects
        result = {}
        for group_chat_id, stats in stats_dict.items():
            # Convert chat_identifiers_seen set to sorted list
            stats["chat_identifiers_seen"] = sorted(list(stats["chat_identifiers_seen"]))

            # Convert participant_stats to ParticipantActivityStatsRecord objects
            participant_stats_records = {}
            for identifier, participant_data in stats["participant_stats"].items():
                # Remove temporary field
                participant_data.pop("_total_message_length", None)
                participant_stats_records[identifier] = ParticipantActivityStatsRecord(**participant_data)

            stats["participant_stats"] = participant_stats_records

            result[group_chat_id] = GroupChatStatsRecord(**stats)

        return result

    def _merge_group_chat_statistics(
            self,
            existing: Optional[GroupChatStatsRecord],
            batch: GroupChatStatsRecord
        ) -> GroupChatStatsRecord:
        """
        Merge batch group chat statistics with existing statistics.

        Args:
            existing: Existing statistics from database (None if first time)
            batch: New statistics computed from batch

        Returns:
            Merged GroupChatStatsRecord
        """
        if existing is None:
            return batch

        # Merge participant stats
        merged_participant_stats = {}

        # Start with existing participant stats
        for identifier, existing_stats in existing.participant_stats.items():
            merged_participant_stats[identifier] = {
                "message_count": existing_stats.message_count,
                "avg_message_length": existing_stats.avg_message_length,
                "hourly_distribution_utc": existing_stats.hourly_distribution_utc.copy(),
                "first_message_timestamp": existing_stats.first_message_timestamp,
                "last_message_timestamp": existing_stats.last_message_timestamp,
                "_total_message_length": existing_stats.avg_message_length * existing_stats.message_count,
            }

        # Merge in batch participant stats
        for identifier, batch_stats in batch.participant_stats.items():
            if identifier not in merged_participant_stats:
                # New participant
                merged_participant_stats[identifier] = {
                    "message_count": batch_stats.message_count,
                    "avg_message_length": batch_stats.avg_message_length,
                    "hourly_distribution_utc": batch_stats.hourly_distribution_utc.copy(),
                    "first_message_timestamp": batch_stats.first_message_timestamp,
                    "last_message_timestamp": batch_stats.last_message_timestamp,
                }
            else:
                # Existing participant - merge stats
                participant_data = merged_participant_stats[identifier]

                # Add message counts
                old_count = participant_data["message_count"]
                new_count = batch_stats.message_count
                total_count = old_count + new_count

                # Merge message length average
                old_total_length = participant_data["_total_message_length"]
                new_total_length = batch_stats.avg_message_length * new_count
                participant_data["avg_message_length"] = (old_total_length + new_total_length) / total_count

                participant_data["message_count"] = total_count

                # Merge hourly distributions (element-wise addition)
                for i in range(168):
                    participant_data["hourly_distribution_utc"][i] += batch_stats.hourly_distribution_utc[i]

                # Update first/last message timestamps
                if batch_stats.first_message_timestamp:
                    if participant_data["first_message_timestamp"] is None:
                        participant_data["first_message_timestamp"] = batch_stats.first_message_timestamp
                    else:
                        participant_data["first_message_timestamp"] = min(
                            participant_data["first_message_timestamp"],
                            batch_stats.first_message_timestamp
                        )

                if batch_stats.last_message_timestamp:
                    if participant_data["last_message_timestamp"] is None:
                        participant_data["last_message_timestamp"] = batch_stats.last_message_timestamp
                    else:
                        participant_data["last_message_timestamp"] = max(
                            participant_data["last_message_timestamp"],
                            batch_stats.last_message_timestamp
                        )

        # Convert participant stats back to ParticipantActivityStatsRecord objects
        final_participant_stats = {}
        for identifier, data in merged_participant_stats.items():
            data.pop("_total_message_length", None)  # Remove temporary field
            final_participant_stats[identifier] = ParticipantActivityStatsRecord(**data)

        # Merge top-level fields
        merged_data = {
            "group_chat_id": batch.group_chat_id,
            "participant_identifiers": batch.participant_identifiers,  # Use batch (should be same)
            "group_chat_name": batch.group_chat_name,  # Use batch (most recent)
            "chat_identifiers_seen": sorted(list(
                set(existing.chat_identifiers_seen) | set(batch.chat_identifiers_seen)
            )),
            "messages_from_me": existing.messages_from_me + batch.messages_from_me,
            "messages_to_me": existing.messages_to_me + batch.messages_to_me,
            "total_messages": existing.total_messages + batch.total_messages,
            # Use the most recent last_message info
            "last_message_timestamp": batch.last_message_timestamp if (
                batch.last_message_timestamp and (
                    existing.last_message_timestamp is None or
                    batch.last_message_timestamp > existing.last_message_timestamp
                )
            ) else existing.last_message_timestamp,
            "last_message_text": batch.last_message_text if (
                batch.last_message_timestamp and (
                    existing.last_message_timestamp is None or
                    batch.last_message_timestamp > existing.last_message_timestamp
                )
            ) else existing.last_message_text,
            "last_message_sender": batch.last_message_sender if (
                batch.last_message_timestamp and (
                    existing.last_message_timestamp is None or
                    batch.last_message_timestamp > existing.last_message_timestamp
                )
            ) else existing.last_message_sender,
            "is_last_from_me": batch.is_last_from_me if (
                batch.last_message_timestamp and (
                    existing.last_message_timestamp is None or
                    batch.last_message_timestamp > existing.last_message_timestamp
                )
            ) else existing.is_last_from_me,
            "participant_stats": final_participant_stats,
        }

        return GroupChatStatsRecord(**merged_data)

    async def _sync_messages_to_local_db(
            self,
            messages: List[MessageRecord],
            show_progress: bool = False,
        ) -> int:
        """
        Sync messages to local SQLite database in batches, returning count of successfully synced messages.
        Messages are already sorted oldest to newest, so we maintain that order.
        """
        if not messages:
            return 0

        total_messages = len(messages)
        synced_count = 0
        failed_count = 0

        # Create progress bar if showing progress
        if show_progress:
            with Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TextColumn("•"),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TextColumn("•"),
                TimeElapsedColumn(),
                console=console,
                transient=False
            ) as progress:
                task = progress.add_task("[cyan]Syncing to local DB...", total=total_messages)

                # Process in batches for better performance
                for batch_start in range(0, total_messages, BATCH_SIZE):
                    batch_end = min(batch_start + BATCH_SIZE, total_messages)
                    batch = messages[batch_start:batch_end]

                    try:
                        batch_synced = await sync_batch_to_local_db(batch)
                        synced_count += batch_synced
                        progress.update(task, advance=batch_synced)

                        # Compute and upsert message statistics for this batch
                        try:
                            batch_stats = self._compute_statistics_from_messages(batch)
                            if batch_stats:
                                identifiers = list(batch_stats.keys())
                                existing_stats = await fetch_existing_statistics(identifiers)

                                merged_stats = {}
                                for identifier, batch_stat in batch_stats.items():
                                    existing_stat = existing_stats.get(identifier)
                                    merged_stats[identifier] = self._merge_statistics(existing_stat, batch_stat)

                                await upsert_statistics(merged_stats)
                                log.debug(f"Upserted statistics for {len(merged_stats)} identifiers")
                        except Exception as stats_error:
                            log.error(f"Failed to compute/upsert message statistics for batch: {stats_error}")

                        # Compute and upsert group chat statistics for this batch
                        try:
                            batch_group_stats = self._compute_group_chat_statistics_from_messages(batch)
                            if batch_group_stats:
                                group_chat_ids = list(batch_group_stats.keys())
                                existing_group_stats = await fetch_existing_group_chat_statistics(group_chat_ids)

                                merged_group_stats = {}
                                for group_chat_id, batch_stat in batch_group_stats.items():
                                    existing_stat = existing_group_stats.get(group_chat_id)
                                    merged_group_stats[group_chat_id] = self._merge_group_chat_statistics(existing_stat, batch_stat)

                                await upsert_group_chat_statistics(merged_group_stats)
                                log.debug(f"Upserted group chat statistics for {len(merged_group_stats)} group chats")
                        except Exception as group_stats_error:
                            log.error(f"Failed to compute/upsert group chat statistics for batch: {group_stats_error}")

                    except Exception as e:
                        log.error(f"Failed to sync batch {batch_start}-{batch_end}: {e}")
                        failed_count += len(batch)
                        progress.update(task, advance=len(batch))
        else:
            # Non-interactive mode - no progress bar
            for batch_start in range(0, total_messages, BATCH_SIZE):
                batch_end = min(batch_start + BATCH_SIZE, total_messages)
                batch = messages[batch_start:batch_end]

                try:
                    synced_count += await sync_batch_to_local_db(batch)

                    # Compute and upsert message statistics for this batch
                    try:
                        batch_stats = self._compute_statistics_from_messages(batch)
                        if batch_stats:
                            identifiers = list(batch_stats.keys())
                            existing_stats = await fetch_existing_statistics(identifiers)

                            merged_stats = {}
                            for identifier, batch_stat in batch_stats.items():
                                existing_stat = existing_stats.get(identifier)
                                merged_stats[identifier] = self._merge_statistics(existing_stat, batch_stat)

                            await upsert_statistics(merged_stats)
                            log.debug(f"Upserted statistics for {len(merged_stats)} identifiers")
                    except Exception as stats_error:
                        log.error(f"Failed to compute/upsert message statistics for batch: {stats_error}")

                    # Compute and upsert group chat statistics for this batch
                    try:
                        batch_group_stats = self._compute_group_chat_statistics_from_messages(batch)
                        if batch_group_stats:
                            group_chat_ids = list(batch_group_stats.keys())
                            existing_group_stats = await fetch_existing_group_chat_statistics(group_chat_ids)

                            merged_group_stats = {}
                            for group_chat_id, batch_stat in batch_group_stats.items():
                                existing_stat = existing_group_stats.get(group_chat_id)
                                merged_group_stats[group_chat_id] = self._merge_group_chat_statistics(existing_stat, batch_stat)

                            await upsert_group_chat_statistics(merged_group_stats)
                            log.debug(f"Upserted group chat statistics for {len(merged_group_stats)} group chats")
                    except Exception as group_stats_error:
                        log.error(f"Failed to compute/upsert group chat statistics for batch: {group_stats_error}")

                except Exception as e:
                    log.error(f"Failed to sync batch {batch_start}-{batch_end}: {e}")
                    failed_count += len(batch)

        if failed_count > 0:
            log.warning(f"Failed to sync {failed_count}/{total_messages} messages")

        return synced_count


    async def run_sync(self, show_progress: bool = False):
        """Main sync function with optional progress monitoring"""
        start_time = datetime.now()

        try:
            # Get last synced timestamp and guid
            last_sync, last_guid = await get_last_synced_timestamp()
            if last_sync:
                log.info(f"Last sync: {last_sync}")
            else:
                log.info("No previous sync found, will sync ALL messages from iMessage database")
                log.info("This may take a while on first run...")

            # Fetch new messages
            log.info("Fetching messages from iMessage database...")
            messages = await self._fetch_new_messages(last_sync)

            # Filter out the last synced message to avoid duplicates
            # (Query is inclusive, so we need to remove the last message we already synced)
            if last_guid and messages:
                original_count = len(messages)
                messages = [msg for msg in messages if msg.guid != last_guid]
                if len(messages) < original_count:
                    log.debug(f"Filtered out {original_count - len(messages)} duplicate message(s)")

            if not messages:
                log.info("No new messages to sync")
                return

            # Calculate time range
            oldest_msg = messages[0].timestamp
            newest_msg = messages[-1].timestamp
            time_span = newest_msg - oldest_msg

            log.info(f"Found {len(messages)} new messages")
            log.info(f"Time range: {oldest_msg} to {newest_msg} (span: {time_span})")
            log.info(f"Batch size: {BATCH_SIZE} messages per batch")

            if show_progress:
                estimated_batches = (len(messages) + BATCH_SIZE - 1) // BATCH_SIZE
                log.info(f"Estimated {estimated_batches} batches to process")

            # Sync to local database (messages are already sorted oldest to newest)
            synced = await self._sync_messages_to_local_db(messages, show_progress=show_progress)

            elapsed = datetime.now() - start_time
            log.info(f"Successfully synced {synced}/{len(messages)} messages in {elapsed.total_seconds():.1f}s")

            if synced > 0:
                rate = synced / elapsed.total_seconds()
                log.info(f"Sync rate: {rate:.1f} messages/second")

        except Exception as e:
            elapsed = datetime.now() - start_time
            log.error(f"Error during sync after {elapsed.total_seconds():.1f}s: {e}", exc_info=True)
            raise


class FileLock:
    """Context manager for file-based locking to prevent concurrent runs"""
    def __init__(self, lock_file: Path):
        self.lock_file = lock_file
        self.lock_file.parent.mkdir(parents=True, exist_ok=True)
        self.lock_fd = None

    def __enter__(self):
        self.lock_fd = open(self.lock_file, 'w')
        try:
            # Try to acquire exclusive lock (non-blocking)
            # LOCK_NB means "don't wait" - fail immediately if lock is held
            fcntl.flock(self.lock_fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.lock_fd.write(f"{os.getpid()}\n{datetime.now().isoformat()}\n")
            self.lock_fd.flush()
            return self
        except IOError:
            # Lock is already held by another process
            # Close immediately and don't wait - this prevents pileup
            self.lock_fd.close()
            raise RuntimeError("Another instance is already running")

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.lock_fd:
            fcntl.flock(self.lock_fd.fileno(), fcntl.LOCK_UN)
            self.lock_fd.close()
            # Clean up lock file
            try:
                self.lock_file.unlink()
            except Exception:
                pass


async def main():
    """Entry point"""
    # Check for --verbose flag for manual runs
    verbose = "--verbose" in sys.argv or "-v" in sys.argv

    # Check iMessage DB accessibility
    if not os.path.exists(IMESSAGE_DB_PATH):
        log.error(f"iMessage database not found at {IMESSAGE_DB_PATH}")
        sys.exit(1)

    # Initialize local database
    await init_local_database()

    # Acquire lock to prevent concurrent runs
    # This uses LOCK_NB (non-blocking), so if another instance is running,
    # this will fail immediately rather than waiting in a queue
    try:
        with FileLock(LOCK_FILE):
            log.info("Lock acquired - starting iMessage sync...")

            imessage_logger = iMessageLogger()

            # Show progress in interactive mode
            await imessage_logger.run_sync(show_progress=verbose)

            log.info("Sync complete")

    except RuntimeError as e:
        if "already running" in str(e):
            log.debug("Skipping run - another instance is already running")
            sys.exit(0)  # Exit cleanly, not an error - this is expected behavior
        raise


if __name__ == "__main__":
    asyncio.run(main())
