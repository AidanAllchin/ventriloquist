"""
Main entry point for the project.

File: main.py
Author: Aidan Allchin
Created: 2025-12-23
Last Modified: 2025-12-24
"""

import asyncio
import logging

from src.collection import collect_data
from src.database import init_local_database, store_training_messages
from src.preprocessing import (
    make_training_messages,
    generate_all_training_windows,
    export_windows_to_jsonl,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
log = logging.getLogger(__name__)


async def main():
    """Run the full data collection and preprocessing pipeline."""
    # Initialize database tables
    await init_local_database()

    # Step 1: Collect raw messages from iMessage
    log.info("=== Step 1: Collecting messages from iMessage ===")
    await collect_data()

    # Step 2: Create training messages from contacts
    log.info("\n=== Step 2: Creating training messages ===")
    messages = await make_training_messages()

    # Step 3: Store training messages to database
    log.info("\n=== Step 3: Storing training messages ===")
    await store_training_messages(messages)

    # Step 4: Generate training windows
    log.info("\n=== Step 4: Generating training windows ===")
    window_count = await generate_all_training_windows()

    # Step 5: Export windows to JSONL
    log.info("\n=== Step 5: Exporting training windows ===")
    await export_windows_to_jsonl()

    log.info("\n=== Pipeline complete ===")
    log.info(f"  Training messages: {len(messages):,}")
    log.info(f"  Training windows: {window_count:,}")


if __name__ == "__main__":
    asyncio.run(main())
