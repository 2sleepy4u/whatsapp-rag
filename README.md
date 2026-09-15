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
- [x] Phase 4 — RAG agent with citations (v1)
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
uv run chat-rag index --window-mode mean # windows = pooled message vectors (much faster)
uv run chat-rag index --recreate         # drop and rebuild the collection
uv run chat-rag search "vacanza in Spagna" --top 10
```

Indexing is incremental: re-running only embeds new or changed messages, and
identical texts are embedded once and reused (chat messages repeat a lot).
`--window-mode model` encodes each window with the embedding model (best
quality); `--window-mode mean` averages the member message vectors instead,
which is roughly two orders of magnitude faster and is plenty for clustering.
The progress bar shows items/s, embedding throughput and an ETA.

### Ask (v1)

```bash
ollama pull qwen2.5:7b
uv run chat-rag ask "Quanti messaggi ha inviato Marco nel 2024?"
uv run chat-rag ask "Cosa ci siamo detti sulla vacanza in Spagna?"
uv run chat-rag ask                      # interactive mode
uv run chat-rag ask "..." --show-steps   # show which tools the model called
uv run chat-rag expand <id> --context 3  # full quote with surrounding messages
```

The model decides dynamically which tools to use (semantic search, keyword/FTS,
statistics, surrounding context) and every answer cites real message ids as
`[id]`; the CLI resolves them to snippets and lets you expand the full quote.
Answers stream token-by-token and tool calls are printed as they run, so you
see progress instead of a silent spinner. `Ollama` generation is capped with
`CHAT_RAG_LLM_NUM_PREDICT` and thinking models (qwen3) can be quieted with
`CHAT_RAG_LLM_THINK=0`.

Data lives under `./data/` (gitignored).

## Development

```bash
uv run pytest
uv run python scripts/gen_mock_export.py --messages 178000   # synthetic export -> data/exports/
uv run python scripts/bench_embed.py --model bge-m3          # embedding throughput benchmark
```
