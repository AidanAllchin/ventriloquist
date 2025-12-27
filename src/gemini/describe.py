"""
Attachment description generation orchestrator.

Processes attachments from iMessage database, generates descriptions via
Gemini API, and caches results for use in training data generation.

File: gemini/describe.py
Author: Aidan Allchin
Created: 2025-12-27
Last Modified: 2025-12-27
"""

import asyncio
import json
import logging
import os
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    MofNCompleteColumn,
)
from rich.console import Console

from ..database import (
    cache_description,
    get_cached_descriptions,
    get_processing_stats,
)
from ..models import AttachmentForProcessing
from .client import GeminiClient
from .content_types import get_content_type, should_call_gemini
from .convert import prepare_for_gemini

log = logging.getLogger(__name__)
console = Console()

# iMessage database path
IMESSAGE_DB = Path.home() / "Library" / "Messages" / "chat.db"

# Training contacts file
CONTACTS_FILE = Path("data/contacts_to_ids.jsonl")


def load_training_contacts() -> Tuple[Set[str], Dict[str, str]]:
    """
    Load training contacts from contacts_to_ids.jsonl.

    Returns:
        Tuple of:
        - Set of all identifiers (phone numbers, emails)
        - Dict mapping identifier to contact name
    """
    if not CONTACTS_FILE.exists():
        raise FileNotFoundError(f"Contacts file not found: {CONTACTS_FILE}")

    identifiers = set()
    id_to_name = {}

    with open(CONTACTS_FILE) as f:
        for line in f:
            data = json.loads(line)
            name = data["contact"]
            for identifier in data["ids"]:
                identifiers.add(identifier)
                id_to_name[identifier] = name

    return identifiers, id_to_name


def get_attachments_for_training(
    training_ids: Set[str],
) -> List[AttachmentForProcessing]:
    """
    Query iMessage database for attachments from training contacts.

    Only includes attachments where ALL chat participants are training contacts.

    Args:
        training_ids: Set of training contact identifiers

    Returns:
        List of AttachmentInfo objects
    """
    if not IMESSAGE_DB.exists():
        raise FileNotFoundError(f"iMessage database not found: {IMESSAGE_DB}")

    conn = sqlite3.connect(f"file:{IMESSAGE_DB}?mode=ro", uri=True)

    try:
        # Query to get attachments with their message/chat context
        # This is a complex query that joins attachments to messages to chats
        query = """
        SELECT DISTINCT
            a.guid,
            a.mime_type,
            a.uti,
            a.filename,
            a.transfer_name,
            COALESCE(a.total_bytes, 0) as total_bytes,
            COALESCE(a.is_outgoing, 0) as is_outgoing,
            COALESCE(m.is_audio_message, 0) as is_audio_message,
            c.chat_identifier,
            (
                SELECT GROUP_CONCAT(h2.id, '|')
                FROM chat_handle_join chj2
                JOIN handle h2 ON chj2.handle_id = h2.ROWID
                WHERE chj2.chat_id = c.ROWID
            ) as all_handles
        FROM attachment a
        JOIN message_attachment_join maj ON a.ROWID = maj.attachment_id
        JOIN message m ON maj.message_id = m.ROWID
        JOIN chat_message_join cmj ON m.ROWID = cmj.message_id
        JOIN chat c ON cmj.chat_id = c.ROWID
        WHERE a.filename IS NOT NULL
        """

        cursor = conn.execute(query)
        attachments = []

        for row in cursor:
            (
                guid,
                mime_type,
                uti,
                filename,
                transfer_name,
                total_bytes,
                is_outgoing,
                is_audio_message,
                chat_identifier,
                all_handles,
            ) = row

            # Parse chat members - only use handles, not chat_identifier
            # (chat_identifier for group chats is like "chat123456", not a contact ID)
            chat_members = []
            if all_handles:
                chat_members = [h for h in all_handles.split("|") if h]

            # For DMs without handles, use chat_identifier (which IS the contact)
            if not chat_members and chat_identifier:
                # DMs have chat_identifier like "+1234567890" or "email@example.com"
                # Group chats have chat_identifier like "chat123456789"
                if not chat_identifier.startswith("chat"):
                    chat_members = [chat_identifier]

            # Deduplicate
            chat_members = list(set(chat_members))

            # Filter: ALL participants must be training contacts
            # (Empty list means we couldn't determine members - skip)
            if not chat_members:
                continue

            # Check if ALL members are training contacts
            all_training = all(
                member in training_ids for member in chat_members
            )
            if not all_training:
                continue

            attachments.append(
                AttachmentForProcessing(
                    guid=guid,
                    mime_type=mime_type,
                    uti=uti,
                    filename=filename,
                    transfer_name=transfer_name,
                    total_bytes=total_bytes,
                    is_outgoing=bool(is_outgoing),
                    is_audio_message=bool(is_audio_message),
                    chat_members=chat_members,
                )
            )

        return attachments

    finally:
        conn.close()


def expand_path(filename: Optional[str]) -> Optional[str]:
    """Expand ~ in iMessage attachment paths."""
    if not filename:
        return None
    # iMessage stores paths with ~ prefix
    return os.path.expanduser(filename.replace("~", "~"))


async def describe_attachment(
    client: GeminiClient,
    attachment: AttachmentForProcessing,
) -> Tuple[str, Optional[str], Optional[str]]:
    """
    Generate a description for a single attachment.

    Args:
        client: GeminiClient instance
        attachment: AttachmentInfo to describe

    Returns:
        Tuple of (content_type, description, error)
    """
    content_type = get_content_type(attachment.mime_type, attachment.uti)
    file_path = expand_path(attachment.filename)

    # Check if file exists
    if not file_path or not Path(file_path).exists():
        return content_type, "[NA]", None

    # Determine if we should call Gemini
    if not should_call_gemini(attachment.mime_type, attachment.is_audio_message):
        # For non-Gemini types, use transfer_name or "[file]"
        if content_type == "audio" and attachment.transfer_name:
            # Music file - use original filename
            return content_type, attachment.transfer_name, None
        elif content_type in ("contact", "location"):
            # Parse locally - for now just return placeholder
            # TODO: Implement vCard and location parsing
            return content_type, f"[{content_type}]", None
        else:
            # Generic file - use filename
            name = attachment.transfer_name or Path(file_path).name
            return content_type, name, None

    # Prepare file for Gemini (convert/downsample as needed)
    temp_path = None
    try:
        file_bytes, mime_type, temp_path = prepare_for_gemini(
            file_path,
            attachment.mime_type,
            attachment.is_audio_message,
        )

        # Call appropriate Gemini method
        description = None
        if not mime_type:
            log.warning(f"No MIME type found for {attachment.guid}")
            return content_type, "[NA]", "No MIME type found"

        if content_type == "image":
            if file_bytes:
                description = await client.describe_image(
                    image_bytes=file_bytes, mime_type=mime_type
                )
            else:
                description = await client.describe_image(
                    image_path=temp_path or file_path, mime_type=mime_type
                )

        elif content_type == "video":
            video_path = temp_path or file_path
            description = await client.describe_video(video_path, mime_type)

        elif content_type == "audio":
            audio_path = temp_path or file_path
            description = await client.transcribe_audio(audio_path, mime_type)

        elif content_type == "document":
            description = await client.describe_document(file_path, mime_type)

        if description:
            return content_type, description, None
        else:
            return content_type, "[NA]", "Gemini returned no description"

    except FileNotFoundError:
        return content_type, "[NA]", None
    except Exception as e:
        # Ensure error message is ASCII-safe for logging/caching
        error_msg = str(e).encode("ascii", errors="replace").decode("ascii")
        log.warning(f"Error processing {attachment.guid}: {error_msg}")
        return content_type, None, error_msg
    finally:
        # Clean up temp file
        if temp_path and temp_path != file_path:
            Path(temp_path).unlink(missing_ok=True)


async def process_attachments(
    api_key: Optional[str] = None,
    dry_run: bool = False,
    limit: Optional[int] = None,
) -> Dict[str, int]:
    """
    Process all attachments from training contacts.

    Args:
        api_key: Gemini API key (or use GEMINI_API_KEY env var)
        dry_run: If True, don't call Gemini API, just report what would be done
        limit: Maximum number of attachments to process (for testing)

    Returns:
        Dict with processing statistics
    """
    # Suppress verbose logging from google-genai and HTTP libraries
    for logger_name in [
        "google",
        "google.genai",
        "google.auth",
        "google_genai",
        "httpx",
        "httpcore",
    ]:
        logging.getLogger(logger_name).setLevel(logging.WARNING)

    training_ids, _ = load_training_contacts()
    attachments = get_attachments_for_training(training_ids)

    if limit:
        attachments = attachments[:limit]

    # Check cache for already-processed attachments
    guids = [a.guid for a in attachments]
    cached = await get_cached_descriptions(guids)

    # Filter to unprocessed
    to_process = [a for a in attachments if a.guid not in cached]

    if dry_run:
        # Just count what would be processed
        by_type = {}
        for attachment in to_process:
            ctype = get_content_type(attachment.mime_type, attachment.uti)
            by_type[ctype] = by_type.get(ctype, 0) + 1

        return {
            "total": len(attachments),
            "cached": len(cached),
            "to_process": len(to_process),
            "by_type": by_type,  # type: ignore
        }

    # Initialize Gemini client
    client = GeminiClient(api_key)

    # Process attachments
    stats = {
        "processed": 0,
        "cached_existing": len(cached),
        "success": 0,
        "errors": 0,
        "skipped": 0,
        "by_type": {},
    }

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TextColumn("[green]{task.fields[success]}[/] ok"),
        TextColumn("[red]{task.fields[errors]}[/] err"),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("/"),
        TimeRemainingColumn(),
        console=console,
        transient=False,
    ) as progress:
        task = progress.add_task(
            "Processing attachments",
            total=len(to_process),
            success=0,
            errors=0,
        )

        for attachment in to_process:
            content_type, description, error = await describe_attachment(
                client, attachment
            )

            # Update stats
            stats["processed"] += 1
            stats["by_type"][content_type] = stats["by_type"].get(content_type, 0) + 1

            if error:
                stats["errors"] += 1
            elif description:
                stats["success"] += 1
            else:
                stats["skipped"] += 1

            # Cache the result
            await cache_description(
                attachment_guid=attachment.guid,
                content_type=content_type,
                description=description,
                mime_type=attachment.mime_type,
                file_exists=description != "[NA]",
                error=error,
            )

            # Update progress bar
            progress.update(
                task,
                advance=1,
                success=stats["success"],
                errors=stats["errors"],
            )

    console.print(f"\n[bold green]Processing complete![/]")
    console.print(f"  API requests: {client.request_count}")
    console.print(f"  Success: {stats['success']}, Errors: {stats['errors']}, Skipped: {stats['skipped']}")

    return stats


async def main():
    """CLI entry point."""
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Generate descriptions for iMessage attachments"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't call Gemini API, just report what would be done",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Maximum number of attachments to process",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show current processing statistics and exit",
    )
    parser.add_argument(
        "--clear-errors",
        action="store_true",
        help="Clear cached errors for retry",
    )

    args = parser.parse_args()

    if args.stats:
        stats = await get_processing_stats()
        print(f"Total cached: {stats['total_cached']}")
        print(f"With description: {stats['with_description']}")
        print(f"With error: {stats['with_error']}")
        print(f"Missing files: {stats['missing_files']}")
        print("By type:")
        for ctype, count in sorted(stats["by_type"].items()):
            print(f"  {ctype}: {count}")
        return

    if args.clear_errors:
        from ..database import clear_errors

        cleared = await clear_errors()
        print(f"Cleared {cleared} cached errors")
        return

    stats = await process_attachments(
        api_key=os.getenv("GEMINI_API_KEY"),
        dry_run=args.dry_run,
        limit=args.limit,
    )

    print("\nProcessing complete:")
    print(f"  Processed: {stats.get('processed', 0)}")
    print(f"  Success: {stats.get('success', 0)}")
    print(f"  Errors: {stats.get('errors', 0)}")


if __name__ == "__main__":
    asyncio.run(main())
