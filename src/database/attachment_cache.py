"""
Database cache operations for attachment descriptions.

Stores Gemini-generated descriptions keyed by attachment GUID for reuse
across preprocessing runs.

File: database/attachment_cache.py
Author: Aidan Allchin
Created: 2025-12-27
Last Modified: 2025-12-27
"""

import logging
from datetime import datetime
from typing import List, Optional

import aiosqlite

from ..models import CachedDescription
from .common import LOCAL_DB_PATH

log = logging.getLogger(__name__)


async def get_cached_description(attachment_guid: str) -> Optional[CachedDescription]:
    """
    Get a cached description for an attachment.

    Args:
        attachment_guid: The attachment's GUID

    Returns:
        CachedDescription if found, None otherwise
    """
    async with aiosqlite.connect(LOCAL_DB_PATH) as conn:
        async with conn.execute(
            """
            SELECT attachment_guid, content_type, description, mime_type,
                   file_exists, created_at, error
            FROM attachment_descriptions
            WHERE attachment_guid = ?
            """,
            (attachment_guid,),
        ) as cursor:
            row = await cursor.fetchone()
            if row:
                return CachedDescription(
                    attachment_guid=row[0],
                    content_type=row[1],
                    description=row[2],
                    mime_type=row[3],
                    file_exists=bool(row[4]),
                    created_at=row[5],
                    error=row[6],
                )
    return None


async def get_cached_descriptions(
    attachment_guids: List[str],
) -> dict[str, CachedDescription]:
    """
    Get cached descriptions for multiple attachments.

    Args:
        attachment_guids: List of attachment GUIDs

    Returns:
        Dict mapping GUID to CachedDescription (only includes found entries)
    """
    if not attachment_guids:
        return {}

    results = {}
    # SQLite has a limit of ~999 variables, so batch queries
    batch_size = 500

    async with aiosqlite.connect(LOCAL_DB_PATH) as conn:
        for i in range(0, len(attachment_guids), batch_size):
            batch = attachment_guids[i : i + batch_size]
            placeholders = ",".join("?" * len(batch))

            async with conn.execute(
                f"""
                SELECT attachment_guid, content_type, description, mime_type,
                       file_exists, created_at, error
                FROM attachment_descriptions
                WHERE attachment_guid IN ({placeholders})
                """,
                batch,
            ) as cursor:
                async for row in cursor:
                    results[row[0]] = CachedDescription(
                        attachment_guid=row[0],
                        content_type=row[1],
                        description=row[2],
                        mime_type=row[3],
                        file_exists=bool(row[4]),
                        created_at=row[5],
                        error=row[6],
                    )

    return results


async def cache_description(
    attachment_guid: str,
    content_type: str,
    description: Optional[str],
    mime_type: Optional[str] = None,
    file_exists: bool = True,
    error: Optional[str] = None,
) -> None:
    """
    Cache a description for an attachment.

    Args:
        attachment_guid: The attachment's GUID
        content_type: The content type (image, video, audio, etc.)
        description: The generated description (or None if failed)
        mime_type: The MIME type of the attachment
        file_exists: Whether the file exists on disk
        error: Error message if generation failed
    """
    async with aiosqlite.connect(LOCAL_DB_PATH) as conn:
        await conn.execute(
            """
            INSERT OR REPLACE INTO attachment_descriptions
            (attachment_guid, content_type, description, mime_type, file_exists, created_at, error)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                attachment_guid,
                content_type,
                description,
                mime_type,
                1 if file_exists else 0,
                datetime.now().isoformat(),
                error,
            ),
        )
        await conn.commit()


async def cache_descriptions_batch(
    descriptions: List[tuple],
) -> None:
    """
    Cache multiple descriptions in a single transaction.

    Args:
        descriptions: List of tuples:
            (attachment_guid, content_type, description, mime_type, file_exists, error)
    """
    if not descriptions:
        return

    async with aiosqlite.connect(LOCAL_DB_PATH) as conn:
        now = datetime.now().isoformat()
        await conn.executemany(
            """
            INSERT OR REPLACE INTO attachment_descriptions
            (attachment_guid, content_type, description, mime_type, file_exists, created_at, error)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (guid, ctype, desc, mime, 1 if exists else 0, now, err)
                for guid, ctype, desc, mime, exists, err in descriptions
            ],
        )
        await conn.commit()


async def get_processing_stats() -> dict:
    """
    Get statistics about attachment processing.

    Returns:
        Dict with counts: total, processed, errors, by_type
    """
    stats = {
        "total_cached": 0,
        "with_description": 0,
        "with_error": 0,
        "missing_files": 0,
        "by_type": {},
    }

    async with aiosqlite.connect(LOCAL_DB_PATH) as conn:
        # Total cached
        async with conn.execute(
            "SELECT COUNT(*) FROM attachment_descriptions"
        ) as cursor:
            row = await cursor.fetchone()
            stats["total_cached"] = row[0] if row else 0

        # With description
        async with conn.execute(
            "SELECT COUNT(*) FROM attachment_descriptions WHERE description IS NOT NULL"
        ) as cursor:
            row = await cursor.fetchone()
            stats["with_description"] = row[0] if row else 0

        # With error
        async with conn.execute(
            "SELECT COUNT(*) FROM attachment_descriptions WHERE error IS NOT NULL"
        ) as cursor:
            row = await cursor.fetchone()
            stats["with_error"] = row[0] if row else 0

        # Missing files
        async with conn.execute(
            "SELECT COUNT(*) FROM attachment_descriptions WHERE file_exists = 0"
        ) as cursor:
            row = await cursor.fetchone()
            stats["missing_files"] = row[0] if row else 0

        # By type
        async with conn.execute(
            """
            SELECT content_type, COUNT(*)
            FROM attachment_descriptions
            GROUP BY content_type
            """
        ) as cursor:
            async for row in cursor:
                stats["by_type"][row[0]] = row[1]

    return stats


async def clear_errors() -> int:
    """
    Clear cached entries that have errors (for retry).

    Returns:
        Number of entries cleared
    """
    async with aiosqlite.connect(LOCAL_DB_PATH) as conn:
        cursor = await conn.execute(
            "DELETE FROM attachment_descriptions WHERE error IS NOT NULL"
        )
        await conn.commit()
        return cursor.rowcount
