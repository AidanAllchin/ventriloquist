"""
File: preprocessing/utils.py
Author: Aidan Allchin
Created: 2025-12-23
Last Modified: 2025-12-23

Utility functions for preprocessing.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List

from ..models import Contact

log = logging.getLogger(__name__)


async def load_contacts(filepath: Path = Path("data/contacts_to_ids.jsonl")) -> List[Contact]:
    """
    Load contacts from JSONL file.

    Each line should contain: {"contact": "name", "ids": ["id1", "id2", ...]}

    Args:
        filepath: Path to contacts JSONL file (one contact per line)

    Returns:
        List of Contact objects
    """
    contacts = []
    with open(filepath, "r") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line.strip())
            contacts.append(Contact(name=data["contact"], identifiers=data["ids"]))

    log.info(f"Loaded {len(contacts)} contacts")
    return contacts

def create_identifier_to_contact_map(contacts: List[Contact]) -> Dict[str, str]:
    """
    Create mapping from identifier to contact name.

    Args:
        contacts: List of Contact objects

    Returns:
        Dictionary mapping identifier -> contact name
    """
    mapping = {}
    for contact in contacts:
        for identifier in contact.identifiers:
            mapping[identifier] = contact.name

    log.info(f"Created mapping for {len(mapping)} unique identifiers")
    return mapping
