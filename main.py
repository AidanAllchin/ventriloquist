"""
Main entry point for the project.

File: main.py
Author: Aidan Allchin
Created: 2025-12-23
Last Modified: 2025-12-23
"""

import asyncio
from src.collection import collect_data

if __name__ == "__main__":
    asyncio.run(collect_data())
