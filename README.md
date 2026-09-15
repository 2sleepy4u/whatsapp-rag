# chat-rag

Local, offline assistant for querying and analysing exported WhatsApp chats
(italiano). Text messages are parsed into SQLite; voice notes (phase 2) get
transcribed locally; a local LLM routes natural-language questions to semantic
search, SQL statistics and clustering, and every answer carries verifiable
citations back to the original messages.

## Status

- [x] Phase 0 — tooling (`uv`, Python 3.12)
- [x] Phase 1 — parser + SQLite + FTS5
- [ ] Phase 2 — statistics engine + CLI
- [ ] Phase 3 — embeddings + vector store
- [ ] Phase 4 — RAG agent with citations (v1)
- [ ] Phase 5 — clustering / inside-joke discovery
- [ ] Phase 6 — local web UI
- [ ] Phase 7 — voice-note transcription
- [ ] Phase 8 — NixOS desktop deployment

## Usage

```bash
uv sync
uv run chat-rag import "/path/to/WhatsApp Chat with Marco.txt"
uv run chat-rag info
```

Data lives under `./data/` (gitignored).

## Development

```bash
uv run pytest
```
