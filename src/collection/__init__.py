"""
Collects data from the iMessage database and stores it in a local SQLite database

File: collection/__init__.py
Author: Aidan Allchin
Created: 2025-12-23
Last Modified: 2025-12-23
"""

from .imessage_logger import main as collect_data

__all__ = [
    "collect_data",
]
