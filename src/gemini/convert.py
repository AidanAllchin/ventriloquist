"""
Media format conversion for Gemini API compatibility.

Converts non-standard formats (HEIC, AMR, etc.) to formats Gemini supports.
Also handles downsampling images to 768x768 and trimming audio/video.

File: gemini/convert.py
Author: Aidan Allchin
Created: 2025-12-27
"""

import io
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Tuple

from PIL import Image

log = logging.getLogger(__name__)

# Target dimensions for image downsampling (768x768 = 1 tile = 258 tokens)
MAX_IMAGE_SIZE = 768

# Duration limits
MAX_VIDEO_SECONDS = 15
MAX_AUDIO_SECONDS = 30


def _check_ffmpeg() -> bool:
    """Check if ffmpeg is available on the system."""
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            check=True,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def _check_pillow_heif() -> bool:
    """Check if pillow-heif is available."""
    try:
        import pillow_heif  # noqa: F401

        return True
    except ImportError:
        return False


def downsample_image(image_path: str, max_size: int = MAX_IMAGE_SIZE) -> Tuple[bytes, str]:
    """
    Downsample image to fit within max_size while preserving aspect ratio.

    Args:
        image_path: Path to image file
        max_size: Maximum dimension (width or height)

    Returns:
        Tuple of (image_bytes, mime_type)

    Raises:
        FileNotFoundError: If image doesn't exist
        ValueError: If image format is unsupported
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Handle HEIC separately
    suffix = path.suffix.lower()
    if suffix in (".heic", ".heif"):
        return convert_heic_to_jpeg(image_path, max_size)

    # Load with PIL
    with Image.open(path) as img:
        # Convert to RGB if necessary (handles RGBA, P mode, etc.)
        if img.mode in ("RGBA", "P", "LA"):
            img = img.convert("RGB")
        elif img.mode != "RGB":
            img = img.convert("RGB")

        # Downsample if larger than max_size
        if img.width > max_size or img.height > max_size:
            img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

        # Save to bytes
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=85)
        return buffer.getvalue(), "image/jpeg"


def convert_heic_to_jpeg(
    heic_path: str, max_size: int = MAX_IMAGE_SIZE
) -> Tuple[bytes, str]:
    """
    Convert HEIC image to JPEG bytes.

    Args:
        heic_path: Path to HEIC file
        max_size: Maximum dimension for output

    Returns:
        Tuple of (jpeg_bytes, "image/jpeg")

    Raises:
        ImportError: If pillow-heif not installed
        FileNotFoundError: If file doesn't exist
    """
    if not _check_pillow_heif():
        raise ImportError("pillow-heif required for HEIC conversion: pip install pillow-heif")

    import pillow_heif

    path = Path(heic_path)
    if not path.exists():
        raise FileNotFoundError(f"HEIC file not found: {heic_path}")

    # Read HEIC
    heif_file = pillow_heif.read_heif(str(path))
    image = Image.frombytes(heif_file.mode, heif_file.size, heif_file.data)

    # Convert to RGB if needed
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Downsample
    if image.width > max_size or image.height > max_size:
        image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

    # Save to bytes
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=85)
    return buffer.getvalue(), "image/jpeg"


def trim_video(video_path: str, max_seconds: int = MAX_VIDEO_SECONDS) -> Optional[str]:
    """
    Extract first N seconds of video to a temp file.

    Args:
        video_path: Path to video file
        max_seconds: Maximum duration to extract

    Returns:
        Path to trimmed video file (in temp directory), or None if failed

    Note:
        Caller is responsible for cleaning up the temp file.
    """
    if not _check_ffmpeg():
        log.warning("ffmpeg not available, cannot trim video")
        return None

    path = Path(video_path)
    if not path.exists():
        log.warning(f"Video file not found: {video_path}")
        return None

    # Create temp file with same extension
    suffix = path.suffix or ".mp4"
    temp_file = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    temp_path = temp_file.name
    temp_file.close()

    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",  # Overwrite
                "-i",
                str(path),
                "-t",
                str(max_seconds),
                "-c:v",
                "libx264",  # Re-encode to H.264 for compatibility
                "-preset",
                "fast",
                "-crf",
                "28",  # Good quality, reasonable size
                "-c:a",
                "aac",  # Keep audio, re-encode to AAC
                "-f",
                "mp4",  # Force MP4 container
                temp_path,
            ],
            capture_output=True,
            check=True,
        )
        return temp_path
    except subprocess.CalledProcessError as e:
        log.warning(f"Failed to trim video: {e.stderr.decode()}")
        Path(temp_path).unlink(missing_ok=True)
        return None


def trim_audio(audio_path: str, max_seconds: int = MAX_AUDIO_SECONDS) -> Optional[str]:
    """
    Extract first N seconds of audio and convert to MP3.

    Args:
        audio_path: Path to audio file
        max_seconds: Maximum duration to extract

    Returns:
        Path to trimmed MP3 file (in temp directory), or None if failed

    Note:
        Caller is responsible for cleaning up the temp file.
    """
    if not _check_ffmpeg():
        log.warning("ffmpeg not available, cannot trim audio")
        return None

    path = Path(audio_path)
    if not path.exists():
        log.warning(f"Audio file not found: {audio_path}")
        return None

    temp_file = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
    temp_path = temp_file.name
    temp_file.close()

    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",  # Overwrite
                "-i",
                str(path),
                "-t",
                str(max_seconds),
                "-acodec",
                "libmp3lame",
                "-ab",
                "128k",  # Good quality, reasonable size
                temp_path,
            ],
            capture_output=True,
            check=True,
        )
        return temp_path
    except subprocess.CalledProcessError as e:
        log.warning(f"Failed to trim audio: {e.stderr.decode()}")
        Path(temp_path).unlink(missing_ok=True)
        return None


def convert_amr_to_mp3(amr_path: str) -> Optional[str]:
    """
    Convert AMR audio to MP3.

    Args:
        amr_path: Path to AMR file

    Returns:
        Path to MP3 file (in temp directory), or None if failed

    Note:
        Caller is responsible for cleaning up the temp file.
    """
    if not _check_ffmpeg():
        log.warning("ffmpeg not available, cannot convert AMR")
        return None

    path = Path(amr_path)
    if not path.exists():
        log.warning(f"AMR file not found: {amr_path}")
        return None

    temp_file = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
    temp_path = temp_file.name
    temp_file.close()

    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(path),
                "-acodec",
                "libmp3lame",
                "-ab",
                "128k",
                temp_path,
            ],
            capture_output=True,
            check=True,
        )
        return temp_path
    except subprocess.CalledProcessError as e:
        log.warning(f"Failed to convert AMR: {e.stderr.decode()}")
        Path(temp_path).unlink(missing_ok=True)
        return None


def prepare_for_gemini(
    file_path: str,
    mime_type: Optional[str],
    is_audio_message: bool = False,
) -> Tuple[Optional[bytes], Optional[str], Optional[str]]:
    """
    Prepare a file for Gemini API by converting/downsampling as needed.

    Args:
        file_path: Path to the attachment file
        mime_type: MIME type of the file
        is_audio_message: True if this is a voice memo

    Returns:
        Tuple of (file_bytes, mime_type, temp_file_path)
        - file_bytes: Bytes to send to Gemini (or None if should read from temp_file)
        - mime_type: MIME type for the API call
        - temp_file_path: Path to temp file that needs cleanup (or None)

    Raises:
        FileNotFoundError: If file doesn't exist
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if not mime_type:
        mime_type = "application/octet-stream"

    mime_lower = mime_type.lower()
    temp_path = None

    # Images: downsample and convert to JPEG
    if mime_lower.startswith("image/"):
        try:
            img_bytes, out_mime = downsample_image(file_path)
            return img_bytes, out_mime, None
        except Exception as e:
            log.warning(f"Failed to process image {file_path}: {e}")
            # Fall back to reading raw file
            return path.read_bytes(), mime_type, None

    # Video: trim to 15 seconds
    if mime_lower.startswith("video/"):
        temp_path = trim_video(file_path)
        if temp_path:
            return None, mime_type, temp_path
        # Fall back to raw file (Gemini may reject if too long)
        return None, mime_type, file_path

    # Audio (voice memos only): trim and convert to MP3
    if mime_lower.startswith("audio/") and is_audio_message:
        # AMR needs conversion
        if mime_lower == "audio/amr":
            temp_path = convert_amr_to_mp3(file_path)
            if temp_path:
                # Also trim
                trimmed = trim_audio(temp_path)
                Path(temp_path).unlink(missing_ok=True)
                if trimmed:
                    return None, "audio/mpeg", trimmed
        else:
            temp_path = trim_audio(file_path)
            if temp_path:
                return None, "audio/mpeg", temp_path

        # Fall back to raw file
        return None, mime_type, file_path

    # PDFs and other documents: send as-is
    return None, mime_type, file_path
