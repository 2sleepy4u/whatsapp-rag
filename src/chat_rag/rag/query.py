"""LLM-based query enrichment for retrieval (optional).

These helpers ask the local model for alternative phrasings (query expansion)
and for a hypothetical message that would answer the question (HyDE). They are
best-effort: any failure yields an empty result so retrieval still works.
"""

from __future__ import annotations

import re

from .agent import LLM

_PARAPHRASE_PROMPT = (
    "Riscrivi la domanda dell'utente in {n} modi diversi in italiano, mantenendo "
    "lo stesso significato e usando parole chiave alternative utili per cercare "
    "in una chat WhatsApp. Rispondi SOLO con le domande riscritte, una per riga, "
    "senza numeri, elenchi o spiegazioni.\n\nDomanda: {question}"
)

_HYDE_PROMPT = (
    "Scrivi un breve estratto (1-2 frasi) di conversazione WhatsApp in italiano che "
    "potrebbe contenere la risposta alla domanda dell'utente. Scrivi solo il testo "
    "del messaggio, senza spiegazioni.\n\nDomanda: {question}"
)

_PREFIX_RE = re.compile(r"^\s*(?:[-*\u2022]|\d+[.)\]]?)\s*")


def _clean_line(line: str) -> str:
    line = _PREFIX_RE.sub("", line).strip()
    return line.strip("\"'")


def _chat_text(llm: LLM, prompt: str) -> str:
    try:
        turn = llm.chat([{"role": "user", "content": prompt}], None)
    except Exception:  # noqa: BLE001 - enrichment is best-effort
        return ""
    return turn.content or ""


def expand_queries(llm: LLM, question: str, n: int = 3) -> list[str]:
    """Return up to ``n`` alternative phrasings of ``question`` (deduplicated)."""
    if n <= 0:
        return []
    content = _chat_text(llm, _PARAPHRASE_PROMPT.format(n=n, question=question))
    out: list[str] = []
    seen: set[str] = set()
    original = question.strip().lower()
    for line in content.splitlines():
        candidate = _clean_line(line)
        key = candidate.lower()
        if len(candidate) >= 3 and key != original and key not in seen:
            seen.add(key)
            out.append(candidate)
    return out[:n]


def hyde_document(llm: LLM, question: str) -> str:
    """Return a hypothetical answer/message to embed as an extra query."""
    return _chat_text(llm, _HYDE_PROMPT.format(question=question)).strip()
