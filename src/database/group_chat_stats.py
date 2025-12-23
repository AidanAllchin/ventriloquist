"""
File: database/group_chat_stats.py
Author: Aidan Allchin
Created: 2025-12-23
Last Modified: 2025-12-23
"""

from datetime import datetime, timezone
import logging
import json
from typing import List, Dict

import aiosqlite

from ..models import GroupChatStatsRecord
from .common import LOCAL_DB_PATH

log = logging.getLogger(__name__)

async def fetch_existing_group_chat_statistics(
    group_chat_ids: List[str]
) -> Dict[str, GroupChatStatsRecord]:
    """
    Fetch existing group chat statistics from the local database for given group_chat_ids.

    Args:
        group_chat_ids: List of group_chat_ids to fetch stats for

    Returns:
        Dictionary mapping group_chat_id to GroupChatStatsRecord (empty dict if no stats found)
    """
    if not group_chat_ids:
        return {}

    try:
        result = {}
        async with aiosqlite.connect(LOCAL_DB_PATH) as conn:
            placeholders = ",".join("?" * len(group_chat_ids))
            query = f"""SELECT * FROM group_chat_stats
                        WHERE group_chat_id IN ({placeholders})"""
            async with conn.execute(query, group_chat_ids) as cursor:
                columns = [description[0] for description in cursor.description]
                async for row in cursor:
                    row_dict = dict(zip(columns, row))
                    group_chat_id = row_dict.get("group_chat_id")
                    if not group_chat_id:
                        continue

                    # Parse JSON fields
                    if row_dict.get("participant_identifiers"):
                        row_dict["participant_identifiers"] = json.loads(row_dict["participant_identifiers"])
                    else:
                        row_dict["participant_identifiers"] = []

                    if row_dict.get("chat_identifiers_seen"):
                        row_dict["chat_identifiers_seen"] = json.loads(row_dict["chat_identifiers_seen"])
                    else:
                        row_dict["chat_identifiers_seen"] = []

                    if row_dict.get("participant_stats"):
                        row_dict["participant_stats"] = json.loads(row_dict["participant_stats"])
                    else:
                        row_dict["participant_stats"] = {}

                    result[group_chat_id] = GroupChatStatsRecord.from_db_dict(row_dict)

        return result
    except Exception as e:
        log.error(f"Error fetching existing group chat statistics: {e}")
        return {}

async def upsert_group_chat_statistics(stats: Dict[str, GroupChatStatsRecord]) -> int:
    """
    Upsert group chat statistics to the local database.

    Args:
        stats: Dictionary mapping group_chat_id to GroupChatStatsRecord

    Returns:
        Number of statistics successfully upserted
    """
    if not stats:
        return 0

    try:
        async with aiosqlite.connect(LOCAL_DB_PATH) as conn:
            for group_chat_id, stat in stats.items():
                # Serialize participant_stats
                participant_stats_json = {}
                for identifier, p_stats in stat.participant_stats.items():
                    participant_stats_json[identifier] = {
                        "message_count": p_stats.message_count,
                        "avg_message_length": p_stats.avg_message_length,
                        "hourly_distribution_utc": p_stats.hourly_distribution_utc,
                        "first_message_timestamp": p_stats.first_message_timestamp.isoformat() if p_stats.first_message_timestamp else None,
                        "last_message_timestamp": p_stats.last_message_timestamp.isoformat() if p_stats.last_message_timestamp else None,
                    }

                await conn.execute("""
                    INSERT INTO group_chat_stats (
                        group_chat_id, participant_identifiers, group_chat_name,
                        chat_identifiers_seen, messages_from_me, messages_to_me, total_messages,
                        last_message_timestamp, last_message_text, last_message_sender,
                        is_last_from_me, participant_stats, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(group_chat_id) DO UPDATE SET
                        participant_identifiers = excluded.participant_identifiers,
                        group_chat_name = excluded.group_chat_name,
                        chat_identifiers_seen = excluded.chat_identifiers_seen,
                        messages_from_me = excluded.messages_from_me,
                        messages_to_me = excluded.messages_to_me,
                        total_messages = excluded.total_messages,
                        last_message_timestamp = excluded.last_message_timestamp,
                        last_message_text = excluded.last_message_text,
                        last_message_sender = excluded.last_message_sender,
                        is_last_from_me = excluded.is_last_from_me,
                        participant_stats = excluded.participant_stats,
                        updated_at = excluded.updated_at
                """, (
                    stat.group_chat_id,
                    json.dumps(stat.participant_identifiers),
                    stat.group_chat_name,
                    json.dumps(stat.chat_identifiers_seen),
                    stat.messages_from_me,
                    stat.messages_to_me,
                    stat.total_messages,
                    stat.last_message_timestamp.isoformat() if stat.last_message_timestamp else None,
                    stat.last_message_text,
                    stat.last_message_sender,
                    1 if stat.is_last_from_me else 0 if stat.is_last_from_me is not None else None,
                    json.dumps(participant_stats_json),
                    datetime.now(timezone.utc).isoformat()
                ))
            await conn.commit()
        return len(stats)
    except Exception as e:
        log.error(f"Error upserting group chat statistics: {e}")
        return 0
