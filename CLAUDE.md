# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Ventriloquist is a personality modeling system that fine-tunes LLMs on iMessage conversations. The goal is to train a completion model that predicts what specific people would say, capturing their tone, humor, and conversational patterns.

**Key insight**: This is a completion model (not instruct-tuned). Identity is conditioned via the training format (`Name: message`), not prompt engineering.

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
- **`src/preprocessing/`**: Transforms raw messages into training format with session-based windowing
- **`src/models/`**: Pydantic models for type-safe data handling
- **`src/training/`**: LoRA training code

### Key Constants

- `SESSION_GAP_THRESHOLD = 2 hours` - Time gap that starts a new conversation session
- `MIN_CONTEXT_MESSAGES = 25` - Minimum messages per training window (pulls from previous sessions)
- `MAX_MESSAGES_PER_WINDOW = 200` - Sanity cap on window size

### Training Data Format

```
DM | Aidan Allchin, Contact Name
---
Aidan Allchin: hey how are you
Contact Name: doing well, you?
Aidan Allchin: [replying to "doing well, you?"] great thanks!
```

Group chats use `Group | participant1, participant2, ...` header.

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
- **Completion model**: Single LoRA learns entire social graph; identity via `Name:` prefix
- **Session windowing**: 2-hour gaps start new sessions; small sessions get context from previous sessions
- **No timestamps in output**: Sequence encodes order; temporal gaps injected at inference via markers like `--- hours later ---`