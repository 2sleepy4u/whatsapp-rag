# chat-rag

Local, offline assistant for querying and analysing exported WhatsApp chats
(italiano). Text messages are parsed into SQLite; voice notes (phase 2) get
transcribed locally; a local LLM routes natural-language questions to semantic
search, SQL statistics and clustering, and every answer carries verifiable
citations back to the original messages.

## Status

- [x] Phase 0 — tooling (`uv`, Python 3.12)
- [x] Phase 1 — parser + SQLite + FTS5
- [x] Phase 2 — statistics engine + CLI
- [x] Phase 3 — embeddings + vector store (Chroma)
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
uv run chat-rag preview "/path/to/WhatsApp Chat with Marco.txt"   # dry-run, no DB writes
uv run chat-rag stats per-sender --chat "Marco"
uv run chat-rag stats response-time --from 2024-01-01 --to 2024-12-31
uv run chat-rag stats words --top 50
uv run chat-rag stats emojis
uv run chat-rag stats sessions --gap 3600
uv run chat-rag stats time-of-day
uv run chat-rag stats volume --bucket month
```

### Embeddings (needs a running Ollama)

```bash
ollama pull bge-m3          # multilingual embeddings
uv run chat-rag index --dry-run          # report what would be embedded
uv run chat-rag index                    # embed messages + conversation windows
uv run chat-rag index --recreate         # drop and rebuild the collection
uv run chat-rag search "vacanza in Spagna" --top 10
```

Indexing is incremental: re-running only embeds new or changed messages. Both a
`message` vector per message and sliding `window` vectors (for topic clustering)
are stored in `data/chroma`.

Data lives under `./data/` (gitignored).

## Development

```bash
uv run pytest
```
