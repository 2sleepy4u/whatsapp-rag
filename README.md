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
- [x] Phase 5 — clustering / inside-joke discovery
- [x] Phase 6 — local web UI
- [x] Phase 7 — voice-note transcription
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

Set `CHAT_RAG_EMBED_CONTEXT=prevnext` to embed each message together with its
previous/next message (short chat messages gain conversational context without
extra model calls); the stored text stays the clean, quotable message.
Switching context mode changes the per-message hash, so a re-run re-embeds
automatically; use `--recreate` when you also want to refresh window vectors.

### Ask (v1)

```bash
ollama pull qwen2.5:7b
uv run chat-rag ask "Quanti messaggi ha inviato Marco nel 2024?"
uv run chat-rag ask "Cosa ci siamo detti sulla vacanza in Spagna?"
uv run chat-rag ask                      # interactive mode
uv run chat-rag ask "..." --show-steps   # show which tools the model called
uv run chat-rag expand <id> --context 3  # full quote with surrounding messages
```

The model decides dynamically which tools to use and every answer cites real
message ids as `[id]`; the CLI resolves them to snippets and lets you expand the
full quote. Retrieval tools, in order of quality:

- `smart_search` — expands the question into paraphrases (and optionally a
  hypothetical answer) and fuses semantic + keyword results (best recall);
- `hybrid_search` — semantic + exact keyword, fused with RRF;
- `window_search` — searches whole conversation windows and returns the real
  messages inside each hit (good for "how did this discussion develop");
- `semantic_search` / `keyword_search` — single-channel lookups;
- searches accept `context=N` to also return the N real messages before/after
  each hit, so the model sees the surrounding exchange without extra calls.

Answers stream token-by-token and tool calls are printed as they run, so you
see progress instead of a silent spinner. `Ollama` generation is capped with
`CHAT_RAG_LLM_NUM_PREDICT` and sampled with `CHAT_RAG_LLM_TEMPERATURE` (0 =
deterministic/factual; 0.3-0.7 for more variety). `CHAT_RAG_LLM_THINK` is
tri-state for thinking models (qwen3): empty leaves the model default, `1`
enables reasoning, `0` disables it. Set `CHAT_RAG_SYSTEM_PROMPT_FILE` to swap in
your own system prompt (`{today}` and `{chats}` are interpolated; see
`prompts/analyst.txt`).

### Topics & inside jokes

```bash
uv run chat-rag topics --min-size 5 --evolution month   # cluster windows into themes
uv run chat-rag jokes --min-count 5                     # repeated-phrase candidates
uv run chat-rag timeline "la papera" --bucket month     # how a phrase evolved
```

`topics` normalises window embeddings, reduces them with PCA and clusters with
HDBSCAN, labelling each cluster with its most distinctive terms (c-TF-IDF) and
central messages (usable as citations). `jokes` counts repeated 2–4 word
n-grams, suppresses fragments subsumed by longer phrases, and reports frequency,
date span and main users. All three are also exposed to the agent as
`topic_clusters`, `inside_joke_candidates` and `phrase_timeline`.

Data lives under `./data/` (gitignored).

### Web UI

```bash
uv run chat-rag serve                 # http://127.0.0.1:8000
uv run chat-rag serve --port 8080
```

A single-page UI (no build step): pick a chat, ask, and watch the answer stream
token-by-token while tool calls are logged. Citations are shown underneath and
each one can be expanded to the surrounding messages. It binds to `127.0.0.1`
by default — there is no auth, so keep it local. The same logic is available
programmatically via `POST /api/ask` and `GET /api/ask/stream` (SSE).

### Voice notes (transcription)

Only works when your export includes the media files (not `<Media omessi>`).

```bash
# CPU: faster-whisper
uv sync --extra voice
uv run chat-rag transcribe --dry-run          # what can/cannot be found
uv run chat-rag transcribe --model small      # transcribe + make searchable
uv run chat-rag index                         # embed the new transcripts

# GPU: whisper.cpp built with Vulkan (see below)
export CHAT_RAG_WHISPER_CPP_BIN=/path/to/whisper-cli
export CHAT_RAG_WHISPER_CPP_MODEL=/path/to/ggml-medium.bin
uv run chat-rag transcribe --engine whisper.cpp
```

Transcripts are stored in the `transcripts` table and (by default) replace the
placeholder `messages.text`, so voice notes become searchable and citable just
like text. Use `--keep-placeholder` to keep the original placeholder.

**Which engine?** On the RX 6650 XT, whisper.cpp + Vulkan is usually several
times faster than CPU; `faster-whisper small/medium` int8 on the i5-12400 is a
solid fallback. Benchmark both on your own files before a bulk run:

```bash
uv run python scripts/bench_transcribe.py --dir data/exports --limit 5 \
  --cpp-bin /path/to/whisper-cli --cpp-model /path/to/ggml-medium.bin
```

## Development

```bash
uv run pytest
uv run python scripts/gen_mock_export.py --messages 178000   # synthetic export -> data/exports/
uv run python scripts/bench_embed.py --model bge-m3          # embedding throughput benchmark
```

### Evaluating the agent

Compare models, prompts, thinking modes and temperatures on a fixed question
set (needs a running Ollama). The report shows tool usage, how many citations
resolved against the DB, latency and answer length:

```bash
uv run python scripts/eval_rag.py \
  --questions scripts/eval_questions.example.json \
  --prompt prompts/analyst.txt --think auto,on --temperature 0,0.3,0.7 \
  --model qwen3:8b --out report.json
```
