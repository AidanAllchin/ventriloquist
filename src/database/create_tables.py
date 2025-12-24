"""
File: database/create_tables.py
Author: Aidan Allchin
Created: 2025-12-23
Last Modified: 2025-12-23
"""

import aiosqlite
from datetime import datetime
import logging
from pathlib import Path

from .common import LOCAL_DB_PATH

# Configure logging
LOG_DIR = Path(__file__).parent.parent / "logs"
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


async def init_local_database() -> None:
    """Initialize the local SQLite database with required tables."""
    LOCAL_DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    async with aiosqlite.connect(LOCAL_DB_PATH) as conn:
        # Messages table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS text_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                message_id INTEGER NOT NULL,
                guid TEXT NOT NULL,
                text TEXT,
                timestamp TEXT NOT NULL,
                sender_id TEXT,
                recipient_id TEXT,
                is_from_me INTEGER NOT NULL,
                service TEXT NOT NULL,
                chat_identifier TEXT,
                is_group_chat INTEGER NOT NULL DEFAULT 0,
                group_chat_name TEXT,
                group_chat_participants TEXT,  -- JSON array
                has_attachments INTEGER NOT NULL DEFAULT 0,
                is_read INTEGER NOT NULL DEFAULT 0,
                read_timestamp TEXT,
                delivered_timestamp TEXT,
                reply_to_guid TEXT,
                thread_originator_guid TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(guid)
            )
        """)

        # Message stats table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS message_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                identifier TEXT NOT NULL,
                individual_from_me INTEGER NOT NULL DEFAULT 0,
                individual_to_me INTEGER NOT NULL DEFAULT 0,
                group_from_me INTEGER NOT NULL DEFAULT 0,
                group_to_me INTEGER NOT NULL DEFAULT 0,
                total_individual INTEGER NOT NULL DEFAULT 0,
                total_group INTEGER NOT NULL DEFAULT 0,
                total_messages INTEGER NOT NULL DEFAULT 0,
                group_chat_names TEXT,  -- JSON array
                last_message_timestamp TEXT,
                last_message_text TEXT,
                is_last_from_me INTEGER,
                is_awaiting_response INTEGER NOT NULL DEFAULT 0,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(identifier)
            )
        """)

        # Group chat stats table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS group_chat_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                group_chat_id TEXT NOT NULL,
                participant_identifiers TEXT,  -- JSON array
                group_chat_name TEXT NOT NULL,
                chat_identifiers_seen TEXT,  -- JSON array
                messages_from_me INTEGER NOT NULL DEFAULT 0,
                messages_to_me INTEGER NOT NULL DEFAULT 0,
                total_messages INTEGER NOT NULL DEFAULT 0,
                last_message_timestamp TEXT,
                last_message_text TEXT,
                last_message_sender TEXT,
                is_last_from_me INTEGER,
                participant_stats TEXT,  -- JSON object
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(group_chat_id)
            )
        """)

        # Training messages table (processed messages ready for training)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS training_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id TEXT NOT NULL,
                from_contact TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                content TEXT NOT NULL,
                is_group_chat INTEGER NOT NULL,
                chat_members TEXT NOT NULL,  -- JSON array
                reply_to_text TEXT,
                thread_originator_guid TEXT
            )
        """)

        # Training windows table (rendered conversation windows for training)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS training_windows (
                window_id INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id TEXT NOT NULL,
                chat_type TEXT NOT NULL,  -- 'group' or 'dm'
                chat_name TEXT,           -- display name for groups, NULL for DMs
                participants TEXT NOT NULL,  -- JSON array
                transcript TEXT NOT NULL,    -- fully rendered, ready for training
                message_count INTEGER NOT NULL,
                session_start TEXT NOT NULL,
                session_end TEXT NOT NULL
            )
        """)

        # Create indexes for faster queries
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_messages_timestamp ON text_messages(timestamp)")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_messages_chat_identifier ON text_messages(chat_identifier)")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_training_msg_chat_ts ON training_messages(chat_id, timestamp)")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_window_chat ON training_windows(chat_id)")

        await conn.commit()
        log.info(f"Local database initialized at {LOCAL_DB_PATH}")
