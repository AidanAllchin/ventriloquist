"""
Gemini integration for attachment description generation.

This module provides tools for generating descriptions of iMessage attachments
using the Gemini 3.0 Flash API. Descriptions are cached in the local database
and used during training data generation.

Usage:
    # Process all attachments (with caching)
    >>> python -m src.gemini.describe

    # Dry run to see what would be processed
    >>> python -m src.gemini.describe --dry-run

    # View processing stats
    >>> python -m src.gemini.describe --stats

File: gemini/__init__.py
Author: Aidan Allchin
Created: 2025-12-27
"""

from .client import GeminiClient
from .content_types import get_content_type, needs_conversion, should_call_gemini
from .convert import (
    convert_heic_to_jpeg,
    downsample_image,
    prepare_for_gemini,
    trim_audio,
    trim_video,
)
from .describe import process_attachments

__all__ = [
    "GeminiClient",
    "get_content_type",
    "needs_conversion",
    "should_call_gemini",
    "convert_heic_to_jpeg",
    "downsample_image",
    "prepare_for_gemini",
    "trim_audio",
    "trim_video",
    "process_attachments",
]
