"""
MIME type to content_type mapping for training data.

File: gemini/content_types.py
Author: Aidan Allchin
Created: 2025-12-27
"""

from typing import Optional


# MIME types that Gemini can process directly
GEMINI_IMAGE_TYPES = {"image/png", "image/jpeg", "image/webp"}
GEMINI_VIDEO_TYPES = {
    "video/mp4",
    "video/quicktime",
    "video/webm",
    "video/3gpp",
    "video/x-flv",
    "video/mpeg",
    "video/x-ms-wmv",
}
GEMINI_AUDIO_TYPES = {
    "audio/aac",
    "audio/flac",
    "audio/mpeg",
    "audio/mp3",
    "audio/x-m4a",
    "audio/m4a",
    "audio/wav",
    "audio/x-wav",
    "audio/opus",
    "audio/pcm",
    "audio/webm",
}

# MIME types that need conversion before sending to Gemini
CONVERT_TO_JPEG = {"image/heic", "image/heic-sequence", "image/tiff", "image/gif", "image/avif"}
CONVERT_TO_MP3 = {"audio/amr"}

# Types we parse locally instead of calling Gemini
LOCAL_PARSE_TYPES = {"text/vcard", "text/x-vlocation"}

# Types we skip (just use filename)
SKIP_TYPES = {
    "application/zip",
    "application/x-zip-compressed",
    "text/html",
    "text/plain",
    "text/markdown",
    "text/x-python-script",
    "application/x-x509-ca-cert",
    "model/vnd.reality",
}


def get_content_type(mime_type: Optional[str], uti: Optional[str] = None) -> str:
    """
    Map MIME type to training data content_type.

    Args:
        mime_type: MIME type string (e.g., "image/jpeg")
        uti: Uniform Type Identifier (fallback)

    Returns:
        One of: text, image, video, audio, document, file, location, contact
    """
    if not mime_type:
        return "file"

    mime_lower = mime_type.lower()

    # Check specific types first
    if mime_lower in LOCAL_PARSE_TYPES:
        if "vcard" in mime_lower:
            return "contact"
        if "vlocation" in mime_lower:
            return "location"

    # Check UTI for contact/location if MIME didn't match
    if uti:
        uti_lower = uti.lower()
        if "vcard" in uti_lower:
            return "contact"
        if "vlocation" in uti_lower or "location" in uti_lower:
            return "location"

    # Category checks
    if mime_lower.startswith("image/"):
        return "image"
    if mime_lower.startswith("video/"):
        return "video"
    if mime_lower.startswith("audio/"):
        return "audio"
    if mime_lower == "application/pdf":
        return "document"
    if mime_lower in {
        "application/msword",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/vnd.ms-excel",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    }:
        return "document"

    return "file"


def needs_conversion(mime_type: Optional[str]) -> Optional[str]:
    """
    Check if MIME type needs format conversion before Gemini.

    Args:
        mime_type: MIME type string

    Returns:
        Target format ("jpeg" or "mp3") or None if no conversion needed
    """
    if not mime_type:
        return None

    mime_lower = mime_type.lower()

    if mime_lower in CONVERT_TO_JPEG:
        return "jpeg"
    if mime_lower in CONVERT_TO_MP3:
        return "mp3"

    return None


def should_call_gemini(mime_type: Optional[str], is_audio_message: bool = False) -> bool:
    """
    Determine if we should call Gemini API for this attachment.

    Args:
        mime_type: MIME type string
        is_audio_message: True if this is a voice memo (from message.is_audio_message)

    Returns:
        True if Gemini should process this attachment
    """
    if not mime_type:
        return False

    mime_lower = mime_type.lower()
    content_type = get_content_type(mime_lower)

    # Skip local-parse types
    if content_type in ("contact", "location"):
        return False

    # Skip generic files
    if content_type == "file":
        return False

    # For audio: only call Gemini for voice memos
    if content_type == "audio":
        return is_audio_message

    # Images, videos, documents: always process
    return content_type in ("image", "video", "document")
