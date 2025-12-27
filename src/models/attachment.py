"""
Attachment-related models for the Ventriloquist project.

File: models/attachment.py
Author: Aidan Allchin
Created: 2025-12-27
Last Modified: 2025-12-27
"""

from typing import List, Optional

from pydantic import BaseModel, Field


class AttachmentInfo(BaseModel):
    """Information about a message attachment."""

    guid: str = Field(..., description="Unique attachment identifier")
    mime_type: Optional[str] = Field(None, description="MIME type (e.g., image/jpeg)")
    uti: Optional[str] = Field(
        None, description="Uniform Type Identifier (e.g., public.jpeg)"
    )
    filename: Optional[str] = Field(
        None, description="Full path in ~/Library/Messages/Attachments/"
    )
    transfer_name: Optional[str] = Field(
        None, description="Original filename (for audio: song name)"
    )
    is_sticker: bool = Field(False, description="True if this is a sticker")


class AttachmentForProcessing(BaseModel):
    """
    Attachment info enriched with context for Gemini processing.

    Includes chat membership info needed to filter by training contacts.
    """

    guid: str = Field(..., description="Unique attachment identifier")
    mime_type: Optional[str] = Field(None, description="MIME type (e.g., image/jpeg)")
    uti: Optional[str] = Field(None, description="Uniform Type Identifier")
    filename: Optional[str] = Field(
        None, description="Full path in ~/Library/Messages/Attachments/"
    )
    transfer_name: Optional[str] = Field(None, description="Original filename")
    total_bytes: int = Field(0, description="File size in bytes")
    is_outgoing: bool = Field(False, description="True if sent by user")
    is_audio_message: bool = Field(
        False, description="True for voice memos (from message.is_audio_message)"
    )
    chat_members: List[str] = Field(
        default_factory=list, description="All phone/email IDs in the chat"
    )


class CachedDescription(BaseModel):
    """A cached attachment description from the database."""

    attachment_guid: str = Field(..., description="Unique attachment identifier")
    content_type: str = Field(
        ..., description="Content type (image, video, audio, etc.)"
    )
    description: Optional[str] = Field(
        None, description="Gemini-generated description or transcription"
    )
    mime_type: Optional[str] = Field(None, description="MIME type of the attachment")
    file_exists: bool = Field(True, description="Whether the file exists on disk")
    created_at: str = Field(..., description="ISO timestamp when cached")
    error: Optional[str] = Field(None, description="Error message if processing failed")