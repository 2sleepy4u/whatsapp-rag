"""Local web UI for chat-rag.

A tiny FastAPI app exposing:

* ``GET /``                - a single-page UI (vanilla JS, no build step);
* ``GET /api/chats``       - available chats;
* ``GET /api/messages/{id}`` - a message, optionally with surrounding context;
* ``POST /api/ask``        - one-shot question -> answer + citations (JSON);
* ``GET /api/ask/stream``  - same, streamed as Server-Sent Events.

It binds to localhost by default: this is private data, so there is no auth and
no reason to expose it on a network.
"""

from __future__ import annotations

import json
import queue
import threading
from dataclasses import asdict
from typing import Callable

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, StreamingResponse

from ..config import Settings, load_settings
from ..db.db import open_db
from ..embed.index import chroma_client, ensure_collection
from ..embed.ollama_embed import OllamaEmbedder
from ..rag.citations import get_message
from ..rag.service import Answer, answer_question
from ..rag.tools import ToolContext, get_context, list_chats

AskFn = Callable[..., Answer]


def _system_extra(chat: str | None) -> str | None:
    return f"Restrict every tool call to chat_id = {chat!r}." if chat else None


def create_app(settings: Settings | None = None, ask: AskFn | None = None) -> FastAPI:
    settings = settings or load_settings()
    chroma = chroma_client(settings.chroma_dir)

    def make_ctx() -> ToolContext:
        return ToolContext(
            conn=open_db(settings.db_path),
            collection=ensure_collection(chroma, "messages"),
            embedder=OllamaEmbedder(settings.embed_model, settings.ollama_host),
        )

    def default_ask(question: str, chat: str | None, max_steps: int,
                    on_event, on_token) -> Answer:
        ctx = make_ctx()
        try:
            return answer_question(
                settings, question, ctx=ctx, max_steps=max_steps,
                system_extra=_system_extra(chat), on_event=on_event, on_token=on_token,
            )
        finally:
            ctx.conn.close()

    ask_fn: AskFn = ask or default_ask

    app = FastAPI(title="chat-rag", docs_url=None, redoc_url=None)

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return INDEX_HTML

    @app.get("/api/chats")
    def api_chats() -> dict:
        ctx = ToolContext(conn=open_db(settings.db_path), collection=None, embedder=None)
        try:
            return {"chats": list_chats(ctx)}
        finally:
            ctx.conn.close()

    @app.get("/api/messages/{message_id}")
    def api_message(message_id: str, context: int = 0) -> dict:
        conn = open_db(settings.db_path)
        try:
            if context <= 0:
                cit = get_message(conn, message_id)
                if cit is None:
                    raise HTTPException(status_code=404, detail="message not found")
                return asdict(cit)
            data = get_context(
                ToolContext(conn=conn, collection=None, embedder=None),
                message_id, before=context, after=context,
            )
            if "error" in data:
                raise HTTPException(status_code=404, detail=data["error"])
            return data
        finally:
            conn.close()

    @app.post("/api/ask")
    def api_ask(payload: dict) -> dict:
        question = str((payload or {}).get("question", "")).strip()
        if not question:
            raise HTTPException(status_code=400, detail="question is required")
        chat = (payload or {}).get("chat") or None
        max_steps = int((payload or {}).get("max_steps", 6))
        try:
            ans = ask_fn(question=question, chat=chat, max_steps=max_steps,
                         on_event=None, on_token=None)
        except ConnectionError:
            raise HTTPException(status_code=503, detail="Ollama non raggiungibile")
        return _answer_json(ans)

    @app.get("/api/ask/stream")
    def api_ask_stream(
        q: str = Query(..., min_length=1),
        chat: str | None = Query(None),
        max_steps: int = Query(6, ge=1, le=12),
    ) -> StreamingResponse:
        return StreamingResponse(
            _stream(ask_fn, q, chat, max_steps),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return app


def _answer_json(ans: Answer) -> dict:
    return {
        "answer": ans.answer,
        "citations": [asdict(c) for c in ans.citations],
        "tools_used": ans.tools_used,
        "steps": [{"tool": s["tool"], "arguments": s["arguments"]} for s in ans.steps],
    }


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False, default=str)}\n\n"


def _stream(ask_fn: AskFn, question: str, chat: str | None, max_steps: int):
    events: queue.Queue = queue.Queue()

    def on_event(kind: str, payload: dict) -> None:
        events.put({"type": "event", "event": kind, "data": payload})

    def on_token(text: str) -> None:
        events.put({"type": "token", "text": text})

    holder: dict = {}

    def worker() -> None:
        try:
            holder["answer"] = ask_fn(question=question, chat=chat, max_steps=max_steps,
                                      on_event=on_event, on_token=on_token)
        except ConnectionError:
            holder["error"] = "Ollama non raggiungibile: avvia `ollama serve` e riprova."
        except Exception as exc:  # noqa: BLE001 - report to the browser
            holder["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            events.put(None)

    threading.Thread(target=worker, daemon=True).start()

    while True:
        item = events.get()
        if item is None:
            break
        yield _sse(item)

    answer = holder.get("answer")
    if answer is None:
        yield _sse({"type": "error", "error": holder.get("error", "errore sconosciuto")})
    else:
        yield _sse({"type": "done", **_answer_json(answer)})


INDEX_HTML = """<!doctype html>
<html lang="it">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>chat-rag</title>
<style>
  :root { --bg:#0f1115; --panel:#171a21; --fg:#e6e6e6; --dim:#9aa4b2; --accent:#4da3ff; --line:#262b36; }
  * { box-sizing: border-box; }
  body { margin:0; font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
         background:var(--bg); color:var(--fg); }
  header { padding:14px 20px; border-bottom:1px solid var(--line); display:flex; gap:12px; align-items:center; }
  header h1 { font-size:16px; margin:0; font-weight:600; }
  header .sub { color:var(--dim); font-size:12px; }
  main { max-width:900px; margin:0 auto; padding:20px; display:flex; flex-direction:column; gap:14px; }
  .row { display:flex; gap:10px; flex-wrap:wrap; align-items:center; }
  select, input, textarea, button { background:var(--panel); color:var(--fg); border:1px solid var(--line);
         border-radius:8px; padding:8px 10px; font:inherit; }
  textarea { width:100%; min-height:64px; resize:vertical; }
  button { cursor:pointer; border-color:var(--accent); color:var(--accent); }
  button:disabled { opacity:.5; cursor:default; }
  .muted { color:var(--dim); font-size:12px; }
  #answer { background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:14px;
            white-space:pre-wrap; line-height:1.5; min-height:60px; }
  #tools { font-size:12px; color:var(--dim); display:flex; flex-direction:column; gap:2px; }
  .cite { background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:10px; margin-top:8px; }
  .cite .meta { color:var(--dim); font-size:12px; margin-bottom:4px; font-family:ui-monospace,monospace; }
  .cite .text { white-space:pre-wrap; }
  .cite button { padding:2px 8px; font-size:12px; margin-top:6px; }
  h2 { font-size:13px; text-transform:uppercase; letter-spacing:.06em; color:var(--dim); margin:6px 0 0; }
  .err { color:#ff6b6b; }
</style>
</head>
<body>
<header>
  <h1>chat-rag</h1>
  <span class="sub">interroga e analizza le tue chat, in locale</span>
</header>
<main>
  <div class="row">
    <label class="muted">Chat</label>
    <select id="chat"><option value="">Tutte</option></select>
    <label class="muted">Passi</label>
    <input id="steps" type="number" min="1" max="12" value="6" style="width:70px">
  </div>
  <textarea id="q" placeholder="Es. Quali erano i nostri inside joke più ricorrenti?"></textarea>
  <div class="row">
    <button id="send">Chiedi</button>
    <span id="status" class="muted"></span>
  </div>
  <h2>Risposta</h2>
  <div id="answer"></div>
  <div id="tools"></div>
  <h2>Citazioni</h2>
  <div id="citations"><span class="muted">Nessuna ancora.</span></div>
</main>
<script>
const el = (id) => document.getElementById(id);
let es = null;

async function loadChats() {
  try {
    const r = await fetch('/api/chats');
    const data = await r.json();
    const sel = el('chat');
    for (const c of data.chats) {
      const o = document.createElement('option');
      o.value = c.chat_id;
      o.textContent = `${c.chat_id} (${c.messages} msg)`;
      sel.appendChild(o);
    }
  } catch (e) { /* ignore */ }
}
loadChats();

function reset() {
  el('answer').textContent = '';
  el('tools').innerHTML = '';
  el('citations').innerHTML = '';
  el('status').textContent = '';
}

function logTool(text) {
  const d = document.createElement('div');
  d.textContent = text;
  el('tools').appendChild(d);
}

function renderCitations(cits) {
  const box = el('citations');
  if (!cits || !cits.length) { box.innerHTML = '<span class="muted">Nessuna citazione.</span>'; return; }
  box.innerHTML = '';
  for (const c of cits) {
    const div = document.createElement('div');
    div.className = 'cite';
    const meta = document.createElement('div');
    meta.className = 'meta';
    meta.textContent = c.found ? `${c.id} · ${c.date} ${c.time} · ${c.sender}` : `${c.id} · non trovato`;
    const text = document.createElement('div');
    text.className = 'text';
    text.textContent = c.found ? c.text : '(id non presente nel database)';
    const btn = document.createElement('button');
    btn.textContent = 'contesto';
    const ctx = document.createElement('div');
    ctx.className = 'text muted';
    btn.onclick = async () => {
      btn.disabled = true;
      const r = await fetch(`/api/messages/${c.id}?context=3`);
      if (!r.ok) { ctx.textContent = 'non disponibile'; btn.disabled = false; return; }
      const d = await r.json();
      ctx.textContent = d.messages.map(m => `${m.date} ${m.time} ${m.sender}: ${m.text}`).join('\\n');
    };
    div.append(meta, text, btn, ctx);
    box.appendChild(div);
  }
}

function ask() {
  const q = el('q').value.trim();
  if (!q) return;
  reset();
  el('send').disabled = true;
  el('status').textContent = 'sto pensando…';
  const chat = el('chat').value;
  const steps = el('steps').value || 6;
  const url = `/api/ask/stream?q=${encodeURIComponent(q)}&max_steps=${encodeURIComponent(steps)}` +
              (chat ? `&chat=${encodeURIComponent(chat)}` : '');
  es = new EventSource(url);
  es.onmessage = (ev) => {
    const m = JSON.parse(ev.data);
    if (m.type === 'token') {
      el('answer').textContent += m.text;
    } else if (m.type === 'event') {
      if (m.event === 'tool_call') logTool(`→ ${m.data.name}(${JSON.stringify(m.data.arguments)})`);
      else if (m.event === 'tool_result') logTool(`  ← ${m.data.count} risultati`);
    } else if (m.type === 'done') {
      renderCitations(m.citations);
      el('status').textContent = m.tools_used && m.tools_used.length ? `strumenti: ${m.tools_used.join(', ')}` : '';
      finish();
    } else if (m.type === 'error') {
      el('answer').innerHTML = `<span class="err">${m.error}</span>`;
      finish();
    }
  };
  es.onerror = () => { if (es) { el('status').textContent = 'connessione interrotta'; finish(); } };
}

function finish() {
  if (es) { es.close(); es = null; }
  el('send').disabled = false;
}

el('send').onclick = ask;
el('q').addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) ask();
});
</script>
</body>
</html>
"""


app = create_app()
