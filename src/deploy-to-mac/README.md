# iMessage Logger

Syncs your iMessage conversations to a local SQLite database for analysis and machine learning.

## Prerequisites

- macOS with iMessage
- [uv](https://docs.astral.sh/uv/) package manager (install with `curl -LsSf https://astral.sh/uv/install.sh | sh`)

## Quick Start

### 1. Install Dependencies

From the project root:

```bash
uv sync
```

This installs all dependencies for the entire project (deploy, preprocessing, and training).

### 2. Grant Full Disk Access

macOS requires Full Disk Access to read the iMessage database.

1. Open **System Settings** > **Privacy & Security** > **Full Disk Access**
2. Click the **+** button
3. Navigate to `/Users/YOUR_USERNAME/Desktop/Projects/message-predictor/.venv/bin/python`
4. Add it to the list and restart Terminal

**Tip:** You may need to show hidden files (Cmd+Shift+.) to see the `.venv` folder.

### 3. Run the Logger

From the project root, run:

```bash
# With interactive progress bar (recommended for first run)
uv run python src/deploy-to-mac/simple_imessage_logger.py --interactive

# Or without progress bar (faster for automated runs)
uv run python src/deploy-to-mac/simple_imessage_logger.py
```

**First run note:** The initial sync will process all your message history, which may take a few minutes depending on how many messages you have.

## What Gets Synced

The logger extracts and stores:

- **Message text** and metadata (timestamp, sender, recipient, service type)
- **Conversation context** (individual vs group chats, participants, names)
- **Message state** (read status, delivery status, attachments)
- **Thread information** (replies, thread originators)
- **Statistics** (per-contact and per-group-chat metrics)

All data is stored in `data/messages.db` (SQLite database at project root).

## Database Schema

### `messages_text_messages`

Complete message records with all metadata.

| Column                  | Type    | Description                        |
| ----------------------- | ------- | ---------------------------------- |
| id                      | INTEGER | Primary key                        |
| message_id              | INTEGER | iMessage ROWID (device-specific)   |
| guid                    | TEXT    | Globally unique message ID         |
| text                    | TEXT    | Message content                    |
| timestamp               | TEXT    | ISO8601 timestamp                  |
| sender_id               | TEXT    | Sender phone/email (normalized)    |
| recipient_id            | TEXT    | Recipient phone/email (normalized) |
| is_from_me              | INTEGER | 1 if sent, 0 if received           |
| service                 | TEXT    | iMessage, SMS, or RCS              |
| chat_identifier         | TEXT    | Internal chat ID                   |
| is_group_chat           | INTEGER | 1 for group chats                  |
| group_chat_name         | TEXT    | User-visible chat name             |
| group_chat_participants | TEXT    | JSON array of participant IDs      |
| has_attachments         | INTEGER | 1 if message has attachments       |
| is_read                 | INTEGER | Read status                        |
| read_timestamp          | TEXT    | When message was read              |
| delivered_timestamp     | TEXT    | When message was delivered         |
| reply_to_guid           | TEXT    | GUID of message this replies to    |
| thread_originator_guid  | TEXT    | GUID of original message in thread |

### `messages_message_stats`

Per-contact statistics computed during sync:

- Message counts (individual/group, sent/received)
- Last message info (timestamp, preview, direction)
- Group chat participation

### `messages_group_chat_stats`

Per-group-chat statistics including:

- Participant activity metrics
- Message counts per participant
- Hourly distribution patterns (UTC)
- Last message info

## Accessing the Data

The database is at `data/messages.db` (project root).

### Using SQLite CLI

```bash
sqlite3 data/messages.db

# Example queries
SELECT COUNT(*) FROM messages_text_messages;
SELECT * FROM messages_message_stats ORDER BY total_messages DESC LIMIT 10;
SELECT * FROM messages_group_chat_stats;
```

### Using Python

```python
import sqlite3

conn = sqlite3.connect('data/messages.db')
cursor = conn.cursor()

# Get all messages from a specific contact
cursor.execute("""
    SELECT timestamp, text, is_from_me
    FROM messages_text_messages
    WHERE sender_id = ? OR recipient_id = ?
    ORDER BY timestamp
""", ('+15551234567', '+15551234567'))

for row in cursor.fetchall():
    print(row)
```

## How It Works

1. **Incremental sync**: Only new messages are processed after the first run
2. **File locking**: Prevents concurrent runs from conflicting
3. **Phone normalization**: Phone numbers are normalized to E.164 format (+15551234567)
4. **Group chat identification**: Group chats are uniquely identified by their participant set, not their name (handles renames correctly)
5. **Statistics computation**: Contact and group stats are computed and updated with each sync

## Automation Options

If you want the logger to run automatically, you have several options:

### Option 1: Cron Job

Run every 5 minutes:

```bash
# Edit crontab
crontab -e

# Add this line (adjust path to your project)
*/5 * * * * cd /Users/YOUR_USERNAME/Desktop/Projects/message-predictor && /Users/YOUR_USERNAME/.local/bin/uv run python src/deploy-to-mac/simple_imessage_logger.py >> logs/cron.log 2>&1
```

### Option 2: While Loop in Background

```bash
# Create a simple runner script
cat > run_logger.sh << 'EOF'
#!/bin/bash
cd /Users/YOUR_USERNAME/Desktop/Projects/message-predictor
while true; do
    uv run python src/deploy-to-mac/simple_imessage_logger.py
    sleep 15
done
EOF

chmod +x run_logger.sh

# Run in background
nohup ./run_logger.sh &
```

### Option 3: macOS Launch Agent

For advanced users, you can create a LaunchAgent plist (see `com.imessage.logger.plist` for reference).

## Troubleshooting

### "unable to open database file" error

- Grant Full Disk Access to the Python executable (`.venv/bin/python`)
- Restart Terminal after granting permissions
- Try restarting your Mac

### Dependencies not found

```bash
# Reinstall dependencies
cd /Users/YOUR_USERNAME/Desktop/Projects/message-predictor
rm -rf .venv
uv sync
```

### No messages syncing

- Verify iMessage database exists: `ls -la ~/Library/Messages/chat.db`
- Check you have messages in the Messages app
- Look for errors in the output when running with `--interactive` flag

### Permission errors with Full Disk Access

- Make sure you added the correct Python binary (`.venv/bin/python`)
- The path must be absolute, not a symlink
- Restart Terminal after granting access
- Try: `ls ~/Library/Messages/chat.db` to verify access

## Development

This logger is part of the Ventriloquist project workspace:

```bash
# Install all workspace dependencies (deploy + preprocessing + training)
uv sync

# Run just the logger dependencies
cd src/deploy-to-mac
uv sync
```

Dependencies are managed via `pyproject.toml`.
