# iMessage Threading Fields: `reply_to_guid` vs `thread_originator_guid`

## Summary

The iMessage database contains two threading-related GUID fields that serve different purposes:

| Field                    | Purpose                             | Coverage          | Useful for Training |
| ------------------------ | ----------------------------------- | ----------------- | ------------------- |
| `reply_to_guid`          | Internal message chain linking      | ~74% of messages  | No                  |
| `thread_originator_guid` | Explicit inline replies & reactions | ~2.8% of messages | Yes                 |

## `reply_to_guid`: Internal Message Chaining

### What It Does

`reply_to_guid` implements a linked-list structure for message ordering within a conversation. Each message points to the previous message in the chat via this field.

### Investigation Findings

Analysis of 770k messages revealed:

- ~74% of messages have `reply_to_guid` set
- ~26% have it NULL

The NULL cases correlate strongly with:

1. **First message** in each conversation (1,973 chats → 93% NULL for first message)
2. **Outgoing messages after 2021** (~60% NULL for messages I sent)

### Year-by-Year Pattern

| Year | My Messages SET | Their Messages SET |
| ---- | --------------- | ------------------ |
| 2020 | 99.9%           | 99.5%              |
| 2021 | 40.2%           | 99.7%              |
| 2022 | 36.2%           | 99.6%              |
| 2023 | 28.0%           | 99.3%              |
| 2024 | 38.3%           | 99.5%              |
| 2025 | 48.0%           | 98.5%              |

The asymmetry reveals this is a technical artifact of Apple's sync/storage implementation, not semantic conversation structure:

- **Incoming messages**: Chain maintained (~99% SET) - the sender's device sets this before transmission
- **Outgoing messages**: Chain broken inconsistently (~60% NULL after 2021) - likely a macOS/iOS behavior change

### Why It's Not Useful for Training

- The NULL pattern doesn't represent topic boundaries or conversation restarts
- It's inconsistent across sent vs received messages
- It changed behavior between 2020 and 2021
- Every message chains to the previous one - this doesn't identify "inline replies"

## `thread_originator_guid`: Explicit Inline Replies

### What It Does

`thread_originator_guid` is set when a user **explicitly** replies to a specific message using iMessage's inline reply feature (long-press $\rightarrow$ Reply).

### Coverage

Only ~2.8% of messages have `thread_originator_guid` set, which matches expected usage patterns - inline replies are intentional user actions, not automatic.

### What Triggers It

1. **Inline replies**: User long-presses a message and selects "Reply"
2. **Tapbacks/Reactions**: "Loved", "Liked", "Laughed", etc. set this field pointing to the reacted message

### Using for Training Data

This is the correct field for the `[replying to "..."]` format in training data:

```
Aidan: [replying to "yo what's the plan tonight"] actually let's do dinner instead
```

Reactions are already formatted with quoted text in the message content:

```
Aidan: Loved "yo what's the plan tonight"
```

These can potentially be wrapped in the same reply syntax for consistency.

## Implementation Notes

### Database Query for Reply Lookups

```python
async def get_message_texts_by_guids(guids: List[str], batch_size: int = 500) -> Dict[str, str]:
    """Batch lookup message text by GUID (for reply resolution)"""
    # SQLite has variable limits, so batch the query
    result = {}
    for i in range(0, len(guids), batch_size):
        batch = guids[i : i + batch_size]
        placeholders = ",".join("?" * len(batch))
        query = f"SELECT guid, text FROM text_messages WHERE guid IN ({placeholders})"
        # ... execute and collect results
    return result
```

### Reply Context Formatting

Per the architecture doc, truncate to ~50 characters:

```python
def format_reply_context(original_text: str, max_len: int = 50) -> Optional[str]:
    if not original_text:
        return None
    truncated = original_text[:max_len]
    if len(original_text) > max_len:
        truncated += "..."
    return truncated
```

## Future Considerations

1. **Reaction handling**: Reactions already contain quoted text. Could normalize format:

   - Current: `Loved "original message"`
   - Possible: `[reacting to "original message"] ❤️` (if emoji tokens desired)

2. **Thread depth**: `thread_originator_guid` points to the root message of a thread, not necessarily the immediate parent in nested reply chains

3. **Cross-chat references**: GUIDs are globally unique, so a message could theoretically reference a message from another chat (though this shouldn't happen in practice)
