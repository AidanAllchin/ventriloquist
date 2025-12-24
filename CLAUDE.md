# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Ventriloquist is a personality modeling system that fine-tunes LLMs on iMessage conversations. The goal is to train a completion model that predicts what specific people would say, capturing their tone, humor, and conversational patterns.

**Key insight**: This is a completion model (not instruct-tuned). Identity is conditioned via JSON structure (`{"name": "...", ...}`), with field names acting as pseudo-special-tokens.

## Commands

```bash
# Run full pipeline (collect → preprocess → store → window → export)
python main.py

# Environment setup
python -m venv .venv
source .venv/bin/activate
pip install -e .
# OR (preferred)
uv run X
```

## Architecture

### Training Data Generation Flow

```
iMessage DB (~/Library/Messages/chat.db)
    ↓
Collection (imessage_logger.py) → messages.db
    ↓
Preprocessing (make_training_messages.py) → TrainingMessage objects
    ↓
Storage (training_data.py) → training_messages table
    ↓
Windowing (generate_windows.py) → training_windows table
    ↓
Export → training_windows.jsonl
```

### Module Purposes

- **`src/collection/`**: Extracts messages from macOS iMessage database with async batch processing
- **`src/database/`**: SQLite operations via aiosqlite. Key tables: `text_messages`, `training_messages`, `training_windows`
- **`src/preprocessing/`**: Transforms raw messages into training format with sliding window generation
- **`src/models/`**: Pydantic models for type-safe data handling
- **`src/training/`**: LoRA training code

### Key Constants

- `WINDOW_SIZE = 100` - Fixed sliding window size (each message gets its own window with up to 99 context messages)

### Training Data Format

**Header (one per window):**
```json
{"type": "dm", "members": ["Aidan Allchin", "Contact Name"], "start": "2024-03-15"}
```

**Messages (one JSON per line):**
```json
{"name": "Aidan Allchin", "delta": "<1m", "content": "hey how are you"}
{"name": "Contact Name", "delta": "<5m", "content": "doing well, you?"}
{"name": "Aidan Allchin", "delta": "<1h", "content": "[replying to \"doing well, you?\"] great thanks!"}
```

**Time delta buckets:** `<1m`, `<5m`, `<1h`, `<12h`, `<1d`, `1d+`

Group chats use `"type": "group"` in header.

### Threading Fields

Two fields exist in iMessage for threading - they are NOT equivalent:
- `thread_originator_guid`: Actual inline replies (~3% of messages) - **USE THIS**
- `reply_to_guid`: Internal message chaining (~74% of messages) - NOT for training

See `docs/imessage_threading_fields.md` for detailed analysis.

## Environment Variables

Required in `.env`:
```
MY_NAME="Your Name"  # Used in training format for your messages
```

Optional:
```
USER_IDENTIFIERS=+1234567890,email@example.com  # Auto-detected if not set
```

## Database Schema

Three main tables:
- `text_messages`: Raw messages with full threading info
- `training_messages`: Preprocessed messages (chat_id, from_contact, timestamp, content, chat_members)
- `training_windows`: Rendered conversation windows ready for training

## Design Decisions

- **Newline splitting**: iMessage stores rapid-fire messages with `\n` separators - these are split into separate TrainingMessages
- **Completion model**: Single LoRA learns entire social graph; identity via `"name":` field in JSON
- **Sliding windows**: Every message gets its own 100-message window (stride 1, maximum overlap)
- **Per-message time deltas**: Each message carries a `delta` field; model learns response timing as personality trait
- **JSON as pseudo-special-tokens**: Field names (`"name":`, `"content":`) provide unambiguous boundaries without custom tokenizer