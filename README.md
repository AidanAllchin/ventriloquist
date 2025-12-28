<div align="center">

<img src="data/ventriloquist.png" alt="Ventriloquist" width="400"/>

</div>

# Ventriloquist

Personality modeling via iMessage fine-tuning. Trains a completion model that predicts what specific people would say, capturing their tone, humor, and conversational patterns.

## Overview

Ventriloquist fine-tunes Qwen3-8B-Base on iMessage texting history using LoRA. The model learns to generate messages as any contact in the chat history, conditioned on conversational context.

**Key insight**: This is a completion model (not instruction-tuned). Identity is encoded via JSON pseudo-special-tokens – the `"name"` field determines who speaks next.

## Quick Start

```bash
# Install dependencies
uv sync

# Run the interactive pipeline menu
uv run python main.py
```

The pipeline has 4 steps:

| Step | Name                    | Requirements              |
| ---- | ----------------------- | ------------------------- |
| 1    | Collect iMessage data   | macOS only                |
| 1.5  | Process attachments     | macOS + Gemini API key    |
| 2    | Preprocess data         | Any platform              |
| 3    | Train model             | GPU recommended           |

```bash
uv run python main.py 1      # Collect
uv run python main.py 1.5    # Process attachments (optional)
uv run python main.py 2      # Preprocess
uv run python main.py 3      # Train
```

### Contacts to IDs Mapping

You must create and populate the `contacts_to_ids.jsonl` file. This contains entries as:

```json
{"contact": "your-contact-name", "ids": ["+12345678900", "me@icloud.com", ...]}
...
```

This is required to map message ids to easily printable message sender names, and is used to aggregate messages from the same person across different IDs into a single thread (sometimes messages are sent or received from iCloud-enabled emails as well as phone numbers).

Note that _your_ contact name and message ID **should not** go in this file. That will be handled by env variables.

## Cross-Platform Workflow

iMessage collection and attachment processing require macOS. Training requires unsloth. Use a Linux/Windows machine with a GPU for step 3.

**On macOS:**

```bash
uv run python main.py 1      # Collect messages
uv run python main.py 1.5    # Process attachments (optional, requires GEMINI_API_KEY)
```

**Transfer data to Linux/Windows:**

```bash
scp -r data/ your-gpu-machine:~/ventriloquist/
```

**On Linux/Windows:**

```bash
uv run python main.py 2
uv run python main.py 3
```

## Pipeline Details

### Step 1: Collection

Extracts messages from macOS iMessage database (`~/Library/Messages/chat.db`).

- Processes both 1:1 and group conversations
- Normalizes phone numbers and emails
- Resolves inline replies via `thread_originator_guid`
- Computes per-contact statistics

**Output:** `data/messages.db`

### Step 1.5: Attachment Processing (Optional)

Generates descriptions for images, videos, and voice memos using Gemini 3.0 Flash.

- **Images**: 1-3 sentence visual descriptions
- **Videos**: Scene and action descriptions (first 15 seconds)
- **Voice memos**: Full transcription
- **Documents**: PDF summaries
- Converts HEIC/Live Photos to JPEG, handles format compatibility
- Caches results in `attachment_descriptions` table

Requires `GEMINI_API_KEY` in `.env`. Skipped attachments get filename as description.

**Supported formats:** JPEG, PNG, HEIC, MP4, MOV, PDF, voice memos (CAF/M4A)

### Step 2: Preprocessing

Transforms raw messages into training-ready format:

1. **Create training messages** — Maps identifiers to contact names, splits newline-separated messages, resolves reply context
2. **Generate windows** — Creates 100-message sliding windows (stride 1)
3. **Export** — Writes to `data/training_windows.jsonl`

**Training data format:**

```json
{"type": "dm", "members": ["Aidan", "John"], "start": "2024-03-15"}
{"name": "Aidan", "delta": "<1m", "reply_to": null, "content_type": "text", "text": "hey what's up"}
{"name": "John", "delta": "<5m", "reply_to": null, "content_type": "image", "text": "A sunset photo over the ocean"}
{"name": "Aidan", "delta": "<1m", "reply_to": "A sunset photo over the ocean", "content_type": "text", "text": "beautiful!"}
{"name": "John", "delta": "<1m", "reply_to": "beautiful!", "content_type": "reaction", "text": "Loved"}
```

**Fields:**

- `name`: Contact who sent the message
- `delta`: Time since previous message (`<1m`, `<5m`, `<1h`, `<12h`, `<1d`, `1d+`)
- `reply_to`: Truncated text of message being replied to, or `null`
- `content_type`: `text`, `image`, `video`, `audio`, `document`, `reaction`, `file`, `contact`, `location`
- `text`: Message content or attachment description

### Step 3: Training

Fine-tunes using Unsloth for 2-5x speedup.

| Setting             | Value                       |
| ------------------- | --------------------------- |
| Base model          | `unsloth/Qwen3-8B-Base`     |
| LoRA rank           | 128                         |
| LoRA alpha          | 256                         |
| Learning rate       | 2.5e-4                      |
| Batch size          | 2x16 gradient accumulation  |
| Epochs              | 2                           |
| Max sequence length | 4096                        |

**Loss masking:** Only the final message of each window contributes to loss. This ensures training matches inference: given full context, predict the next message.

**Output:** `checkpoints/ventriloquist/final/`

Override defaults:

```bash
uv run python -m src.training.train --lora_r 64 --learning_rate 1e-4
```

**Continue training from a checkpoint:**

```bash
# Continue training with modified data or additional epochs
uv run python -m src.training.train --continue_from checkpoints/ventriloquist/final --num_epochs 2

# Resume interrupted training (same data, restores optimizer state)
uv run python -m src.training.train --resume
```

**Quantization for limited VRAM:**

```bash
uv run python -m src.training.train --load_in_8bit  # ~8GB VRAM
uv run python -m src.training.train --load_in_4bit  # ~4GB VRAM
```

## Inference

```bash
uv run python -m src.training.inference

# With quantization for limited VRAM (24GB GPU)
uv run python -m src.training.inference --load_in_8bit

# For 16GB GPUs
uv run python -m src.training.inference --load_in_4bit

# Load specific checkpoint
uv run python -m src.training.inference --adapter_path checkpoints/ventriloquist/checkpoint-1600
```

Commands:

- `/target <name>` — Set who to generate for
- `/members <name1> <name2>` — Set conversation participants
- `/type <dm|group>` — Set chat type
- `/debug` – Show exact prompt and response content
- `/context` — Show conversation history
- `/clear` — Clear conversation history
- `/help` — Show help
- `/quit` — Exit

Type messages as `<name>: <content>`:

```
> John: hey, you free tonight?

  John <5m>: hey, you free tonight?
  Aidan <5m>: yeah what's up
```

## Evaluation

### Standard Metrics

```bash
uv run python -m src.training.evaluate
```

Computes:

- Perplexity on held-out windows
- JSON validity rate
- Delta validity rate
- Per-contact analysis (with `--per_contact`)

### Compare Specific Windows

View expected vs generated output for a specific eval window:

```bash
# Compare window at index 5
uv run python -m src.training.evaluate --compare 5

# Override target speaker
uv run python -m src.training.evaluate --compare 5 --target "John"

# Compare fine-tuned vs base model
uv run python -m src.training.evaluate --compare 5 --with_base
```

Example output:

```
============================================================
Chat: dm with ['Aidan', 'John']
Context: 15 messages
Last message: John: you around?
------------------------------------------------------------
Target: Aidan
------------------------------------------------------------
EXPECTED [<5m>]: yeah what's up
------------------------------------------------------------
FINETUNED [<5m>]: yeah I'm here, what's good
------------------------------------------------------------
BASE (no formatting): I am available for assistance. How may I help you today?
============================================================
```

## Environment Setup

Create `.env` in project root:

```bash
# Required to populate your contact name
MY_NAME="Your Name"

# Optional
USER_IDENTIFIERS=+1234567890,you@email.com  # Auto-detected if not set
GEMINI_API_KEY=your_key_here                 # For attachment processing (step 1.5)
WANDB_API_KEY=your_key_here                  # Or run `wandb login`
```

## Project Structure

```
├── main.py                    # Interactive CLI
├── src/
│   ├── collection/            # iMessage extraction (macOS)
│   │   └── imessage_logger.py
│   ├── gemini/                # Attachment processing
│   │   ├── describe.py        # Orchestrator for attachment descriptions
│   │   ├── client.py          # Gemini API client with rate limiting
│   │   ├── convert.py         # Format conversion (HEIC, video trim, etc.)
│   │   └── content_types.py   # MIME type mapping
│   ├── preprocessing/         # Training data generation
│   │   ├── make_training_messages.py
│   │   └── generate_windows.py
│   └── training/              # Model training & inference
│       ├── train.py           # Main training script
│       ├── inference.py       # Interactive generation
│       ├── evaluate.py        # Metrics & comparison
│       ├── config.py          # Hyperparameters
│       └── data.py            # Dataset & loss masking
├── data/
│   ├── contacts_to_ids.jsonl  # Contact names -> messaging IDs (user-made)
│   ├── messages.db            # Local message database
│   └── training_windows.jsonl # Training data
└── checkpoints/               # Saved models
```

## Design Decisions

**Completion model, not instruct-tuned:** Identity is encoded in the data format (`"name"` field). A single LoRA learns the entire social graph so no per-contact adapters are needed.

**JSON as pseudo-special-tokens:** Field names (`"name":`, `"content":`, `"delta":`) provide unambiguous boundaries without custom tokenizer modifications.

**Sliding windows with stride 1:** Every message becomes the prediction target of its own 50-message window. Maximum overlap reinforces consistent personality modeling.

**Per-message time deltas:** The model learns response timing as a personality trait. Some contacts reply in seconds; others take days.

## Requirements

- Python 3.11+
- macOS for collection (iMessage access)
- GPU with 24GB+ VRAM recommended for training (requires quantization)
- ~10GB disk space for model weights

## Installation

**Basic install (inference on any platform including Mac):**

```bash
uv sync
```

**With training support (requires NVIDIA/AMD/Intel GPU):**

```bash
uv sync --extra train
```

The base installation uses standard HuggingFace `transformers` + `peft` for inference, which works on CPU, CUDA, and Apple Silicon (MPS). Training uses Unsloth for 2-5x speedup but requires a compatible GPU.
