"""
File: database/message_stats.py
Author: Aidan Allchin
Created: 2025-12-23
Last Modified: 2025-12-23
"""

from datetime import datetime, timezone
import logging
import json
from typing import List, Dict

import aiosqlite

from ..models import MessageStatsRecord
from .common import LOCAL_DB_PATH

log = logging.getLogger(__name__)

async def fetch_existing_statistics(identifiers: List[str]) -> Dict[str, MessageStatsRecord]:
    """
    Fetch existing statistics from the local database for given identifiers.

    Args:
        identifiers: List of identifiers to fetch stats for

    Returns:
        Dictionary mapping identifier to MessageStatsRecord (empty dict if no stats found)
    """
    if not identifiers:
        return {}

    try:
        result = {}
        async with aiosqlite.connect(LOCAL_DB_PATH) as conn:
            placeholders = ",".join("?" * len(identifiers))
            query = f"""SELECT * FROM message_stats
                        WHERE identifier IN ({placeholders})"""
            async with conn.execute(query, identifiers) as cursor:
                columns = [description[0] for description in cursor.description]
                async for row in cursor:
                    row_dict = dict(zip(columns, row))
                    identifier = row_dict.get("identifier")
                    if not identifier:
                        continue

                    # Parse JSON fields
                    if row_dict.get("group_chat_names"):
                        row_dict["group_chat_names"] = json.loads(row_dict["group_chat_names"])
                    else:
                        row_dict["group_chat_names"] = []

                    # Parse datetime
                    if row_dict.get("last_message_timestamp"):
                        row_dict["last_message_timestamp"] = datetime.fromisoformat(row_dict["last_message_timestamp"])

                    # Convert boolean fields
                    row_dict["is_awaiting_response"] = bool(row_dict.get("is_awaiting_response", 0))
                    if row_dict.get("is_last_from_me") is not None:
                        row_dict["is_last_from_me"] = bool(row_dict["is_last_from_me"])

                    result[identifier] = MessageStatsRecord(**row_dict)

        return result
    except Exception as e:
        log.error(f"Error fetching existing statistics: {e}")
        return {}

async def upsert_statistics(stats: Dict[str, MessageStatsRecord]) -> int:
    """
    Upsert message statistics to the local database.

    Args:
        stats: Dictionary mapping identifier to MessageStatsRecord

    Returns:
        Number of statistics successfully upserted
    """
    if not stats:
        return 0

    try:
        async with aiosqlite.connect(LOCAL_DB_PATH) as conn:
            for identifier, stat in stats.items():
                await conn.execute("""
                    INSERT INTO message_stats (
                        identifier, individual_from_me, individual_to_me,
                        group_from_me, group_to_me, total_individual, total_group,
                        total_messages, group_chat_names, last_message_timestamp,
                        last_message_text, is_last_from_me, is_awaiting_response, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(identifier) DO UPDATE SET
                        individual_from_me = excluded.individual_from_me,
                        individual_to_me = excluded.individual_to_me,
                        group_from_me = excluded.group_from_me,
                        group_to_me = excluded.group_to_me,
                        total_individual = excluded.total_individual,
                        total_group = excluded.total_group,
                        total_messages = excluded.total_messages,
                        group_chat_names = excluded.group_chat_names,
                        last_message_timestamp = excluded.last_message_timestamp,
                        last_message_text = excluded.last_message_text,
                        is_last_from_me = excluded.is_last_from_me,
                        is_awaiting_response = excluded.is_awaiting_response,
                        updated_at = excluded.updated_at
                """, (
                    stat.identifier,
                    stat.individual_from_me,
                    stat.individual_to_me,
                    stat.group_from_me,
                    stat.group_to_me,
                    stat.total_individual,
                    stat.total_group,
                    stat.total_messages,
                    json.dumps(stat.group_chat_names),
                    stat.last_message_timestamp.isoformat() if stat.last_message_timestamp else None,
                    stat.last_message_text,
                    1 if stat.is_last_from_me else 0 if stat.is_last_from_me is not None else None,
                    1 if stat.is_awaiting_response else 0,
                    datetime.now(timezone.utc).isoformat()
                ))
            await conn.commit()
        return len(stats)
    except Exception as e:
        log.error(f"Error upserting statistics: {e}")
        return 0


async def get_top_contacts(n: int = 10, sort_by: str = "total_messages") -> List[MessageStatsRecord]:
    """
    Retrieve the top N most frequently messaged contacts.

    Args:
        n: Number of contacts to retrieve (default: 10)
        sort_by: Field to sort by - "total_messages", "total_individual", or "total_group" (default: "total_messages")

    Returns:
        List of MessageStatsRecord ordered by message frequency
    """
    valid_sort_fields = {"total_messages", "total_individual", "total_group"}
    if sort_by not in valid_sort_fields:
        raise ValueError(f"Invalid sort_by field. Must be one of: {valid_sort_fields}")

    try:
        result = []
        async with aiosqlite.connect(LOCAL_DB_PATH) as conn:
            query = f"""
                SELECT * FROM message_stats
                ORDER BY {sort_by} DESC
                LIMIT ?
            """
            async with conn.execute(query, (n,)) as cursor:
                columns = [description[0] for description in cursor.description]
                async for row in cursor:
                    row_dict = dict(zip(columns, row))

                    # Parse JSON fields
                    if row_dict.get("group_chat_names"):
                        row_dict["group_chat_names"] = json.loads(row_dict["group_chat_names"])
                    else:
                        row_dict["group_chat_names"] = []

                    # Parse datetime
                    if row_dict.get("last_message_timestamp"):
                        row_dict["last_message_timestamp"] = datetime.fromisoformat(row_dict["last_message_timestamp"])

                    # Convert boolean fields
                    row_dict["is_awaiting_response"] = bool(row_dict.get("is_awaiting_response", 0))
                    if row_dict.get("is_last_from_me") is not None:
                        row_dict["is_last_from_me"] = bool(row_dict["is_last_from_me"])

                    result.append(MessageStatsRecord(**row_dict))

        return result
    except Exception as e:
        log.error(f"Error fetching top contacts: {e}")
        return []
