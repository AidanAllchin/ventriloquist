"""
File: models/contact.py
Author: Aidan Allchin
Created: 2025-12-23
Last Modified: 2025-12-23
"""

from pydantic import BaseModel, Field
from typing import List


class Contact(BaseModel):
    """Lightweight contact model for training data."""

    name: str = Field(..., description="Contact name")
    identifiers: List[str] = Field(..., description="List of phone numbers/emails for this contact")