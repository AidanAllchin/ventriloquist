"""
Common database constants and utilities

File: database/common.py
Author: Aidan Allchin
Created: 2025-12-23
Last Modified: 2025-12-23
"""

from pathlib import Path

DATA_DIR = Path(__file__).parent.parent.parent / "data"
LOCAL_DB_PATH = DATA_DIR / "messages.db"

__all__ = [
    "DATA_DIR",
    "LOCAL_DB_PATH",
]
