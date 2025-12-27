"""
Gemini 3.0 Flash API client for attachment descriptions.

Handles rate limiting, retries, and media upload for generating
descriptions of images, videos, audio, and documents.

File: gemini/client.py
Author: Aidan Allchin
Created: 2025-12-27
Last Modified: 2025-12-27
"""

import asyncio
import logging
import os
import time
from pathlib import Path
from typing import Optional

from google import genai
from google.genai import types

log = logging.getLogger(__name__)

# Rate limiting: 600 req/min = 100ms between requests
MIN_REQUEST_INTERVAL = 0.1  # seconds

# Retry settings
MAX_RETRIES = 3
INITIAL_BACKOFF = 1.0  # seconds

# Model configuration
MODEL_NAME = "gemini-3-flash-preview"

# Prompts for different content types
PROMPTS = {
    "image": (
        "Describe this image in 1-3 concise sentences. Focus on the main subject and action. "
        "Do not start with 'This image shows' or similar phrases. Be direct."
    ),
    "video": (
        "Describe this video in 1-3 concise sentences. Focus on what happens and the setting. "
        "Do not start with 'This video shows' or similar phrases. Be direct."
    ),
    "audio": (
        "Transcribe this voice message. Provide only the spoken words. "
        "If unintelligible, respond with '[unintelligible voice memo]'."
    ),
    "document": (
        "Summarize this document in 1-3 sentences. Focus on the main topic and purpose."
    ),
}


class GeminiClient:
    """
    Client for Gemini 3.0 Flash API with rate limiting and retries.
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Gemini client.

        Args:
            api_key: Gemini API key. If not provided, reads from GEMINI_API_KEY env var.

        Raises:
            ValueError: If no API key provided or found in environment
        """
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Gemini API key required. Set GEMINI_API_KEY environment variable "
                "or pass api_key parameter."
            )

        self.client = genai.Client(api_key=self.api_key)
        self._last_request_time = 0.0
        self._request_count = 0

    async def _rate_limit(self) -> None:
        """Enforce rate limiting between requests."""
        now = time.time()
        elapsed = now - self._last_request_time
        if elapsed < MIN_REQUEST_INTERVAL:
            await asyncio.sleep(MIN_REQUEST_INTERVAL - elapsed)
        self._last_request_time = time.time()

    async def _call_with_retry(
        self,
        contents: list,
    ) -> Optional[str]:
        """
        Call Gemini API with exponential backoff retry.

        Args:
            contents: List of content parts (text, images, files, etc.)

        Returns:
            Generated text response, or None if all retries failed
        """
        backoff = INITIAL_BACKOFF

        for attempt in range(MAX_RETRIES):
            await self._rate_limit()
            self._request_count += 1

            try:
                # Make the API call
                response = await asyncio.to_thread(
                    self.client.models.generate_content,
                    model=MODEL_NAME,
                    contents=contents,
                )

                if response.text:
                    return response.text.strip()
                else:
                    log.warning(f"Empty response from Gemini (attempt {attempt + 1})")
                    return None

            except Exception as e:
                error_str = str(e).lower()

                # Check for rate limiting
                if "resource exhausted" in error_str or "429" in error_str:
                    log.warning(f"Rate limited (attempt {attempt + 1}): {e}")
                    if attempt < MAX_RETRIES - 1:
                        await asyncio.sleep(backoff)
                        backoff *= 2
                    continue

                # Check for invalid request
                if "invalid" in error_str or "400" in error_str:
                    log.error(f"Invalid request to Gemini: {e}")
                    return None

                # Other errors
                log.error(f"Gemini API error (attempt {attempt + 1}): {e}")
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(backoff)
                    backoff *= 2

        log.error(f"All {MAX_RETRIES} retries failed")
        return None

    async def describe_image(
        self,
        image_path: Optional[str] = None,
        image_bytes: Optional[bytes] = None,
        mime_type: str = "image/jpeg",
    ) -> Optional[str]:
        """
        Generate a description of an image.

        Args:
            image_path: Path to image file (mutually exclusive with image_bytes)
            image_bytes: Raw image bytes (mutually exclusive with image_path)
            mime_type: MIME type of the image

        Returns:
            1-3 sentence description, or None if failed
        """
        if image_bytes:
            # Use inline data for bytes
            contents = [
                types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                PROMPTS["image"],
            ]
        elif image_path:
            path = Path(image_path)
            if not path.exists():
                log.warning(f"Image file not found: {image_path}")
                return None
            # Upload the file
            uploaded = await asyncio.to_thread(
                self.client.files.upload, file=str(path)
            )
            contents = [uploaded, PROMPTS["image"]]
        else:
            raise ValueError("Either image_path or image_bytes required")

        return await self._call_with_retry(contents)

    async def describe_video(
        self,
        video_path: str,
        mime_type: str = "video/mp4",
    ) -> Optional[str]:
        """
        Generate a description of a video.

        Args:
            video_path: Path to video file
            mime_type: MIME type of the video

        Returns:
            1-3 sentence description, or None if failed
        """
        path = Path(video_path)
        if not path.exists():
            log.warning(f"Video file not found: {video_path}")
            return None

        # Upload the video file
        uploaded = await asyncio.to_thread(
            self.client.files.upload, file=str(path)
        )

        # Wait for processing (videos need time to process)
        # State is FileState enum, use .name for string comparison
        if not uploaded.state:
            log.warning(f"Video processing failed: {uploaded.name}")
            return None
        
        while uploaded.state.name == "PROCESSING":  # type: ignore
            await asyncio.sleep(1.0)
            uploaded = await asyncio.to_thread(
                self.client.files.get, name=uploaded.name
            )

        if uploaded.state.name != "ACTIVE":  # type: ignore
            log.warning(f"Video processing failed: {uploaded.state.name}")  # type: ignore
            return None

        return await self._call_with_retry([uploaded, PROMPTS["video"]])

    async def transcribe_audio(
        self,
        audio_path: str,
        mime_type: str = "audio/mpeg",
    ) -> Optional[str]:
        """
        Transcribe a voice memo.

        Args:
            audio_path: Path to audio file
            mime_type: MIME type of the audio

        Returns:
            Transcription text, or None if failed
        """
        path = Path(audio_path)
        if not path.exists():
            log.warning(f"Audio file not found: {audio_path}")
            return None

        # Upload the audio file
        uploaded = await asyncio.to_thread(
            self.client.files.upload, file=str(path)
        )

        # Wait for processing
        # State is FileState enum, use .name for string comparison
        if not uploaded.state:
            log.warning(f"Audio processing failed: {uploaded.name}")
            return None
        
        while uploaded.state.name == "PROCESSING":  # type: ignore
            await asyncio.sleep(1.0)
            uploaded = await asyncio.to_thread(
                self.client.files.get, name=uploaded.name
            )

        if uploaded.state.name != "ACTIVE":  # type: ignore
            log.warning(f"Audio processing failed: {uploaded.state.name}")  # type: ignore
            return None

        return await self._call_with_retry([uploaded, PROMPTS["audio"]])

    async def describe_document(
        self,
        doc_path: str,
        mime_type: str = "application/pdf",
    ) -> Optional[str]:
        """
        Summarize a document.

        Args:
            doc_path: Path to document file
            mime_type: MIME type of the document

        Returns:
            1-3 sentence summary, or None if failed
        """
        path = Path(doc_path)
        if not path.exists():
            log.warning(f"Document file not found: {doc_path}")
            return None

        # Upload the document
        uploaded = await asyncio.to_thread(
            self.client.files.upload, file=str(path)
        )

        return await self._call_with_retry([uploaded, PROMPTS["document"]])

    @property
    def request_count(self) -> int:
        """Number of API requests made."""
        return self._request_count
