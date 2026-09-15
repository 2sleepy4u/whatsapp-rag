#!/usr/bin/env python3
"""Generate a deterministic, medium-size Android WhatsApp export for testing.

The output mimics the real Android export format::

    17/04/2023, 14:03 - Marco: Ciao
    17/04/2023, 14:05 - Me: Ehi
    continua qui
    17/04/23, 14:07 - Messaggi e chiamate sono protetti con la crittografia end-to-end.

It is intentionally *synthetic* (Italian filler text, no real people) so it can
be committed / regenerated without touching private data. Timestamps span a
configurable date range with realistic daily/weekly seasonality, and messages
cover the edge cases the parser handles: multi-line bodies, media/voice
placeholders, edited/deleted markers, system notices, calls and duplicates.

Usage::

    uv run python scripts/gen_mock_export.py --messages 150000
"""

from __future__ import annotations

import argparse
import random
from datetime import datetime, timedelta
from pathlib import Path

PREFIX = "WhatsApp Chat with "

GROUP_SENDERS = ["Marco", "Luca", "Giulia", "Sara", "Me"]
PRIVATE_SENDERS = ["Marco", "Me"]
GROUP_NAME = "Amici di sempre"
PRIVATE_NAME = "Marco"

SYSTEM_LINES = [
    "Messaggi e chiamate sono protetti con la crittografia end-to-end.",
    "Marco ha aggiunto Luca",
    "Giulia ha cambiato l'oggetto a \"Vacanze 2024\"",
    "Sara è uscita",
    "Luca ha cambiato l'icona di questo gruppo",
    "Marco ha modificato la descrizione del gruppo",
]

CALL_LINES = [
    "Chiamata vocale",
    "Hai perso una videochiamata",
    "Hai effettuato una chiamata vocale",
    "Videochiamata senza risposta",
]

MEDIA_LINES = [
    "<Media omessi>",
    "IMG-20220417-WA0011.jpg (file allegato)",
    "VID-20230101-WA0002.mp4 (file allegato)",
    "STK-20221224-WA0007.webp (file allegato)",
    "GIF-20201010-WA0003.gif (file allegato)",
    "documento.pdf (file allegato)",
    "<Posizione: 41.9028, 12.4964>",
]

VOICE_LINES = [
    "AUD-20210315-WA0009.opus (file allegato)",
    "AUD-20231102-WA0014.opus (file allegato)",
]

SHORT = [
    "Ciao!", "Ehi, come va?", "Tutto bene dai", "Ci sei?", "Dove siete?",
    "Arrivo tra 10 minuti", "Ok perfetto", "Ahahahah", "Ma dai!", "Che ridere",
    "Sisi ci sto", "Sto arrivando", "Aspetta un attimo", "Domani ti dico",
    "Buongiorno ☀️", "Buonanotte 😴", "Che fame", "Andiamo a prendere un caffè?",
    "Guarda qua", "Non ci credo", "Grande!", "Vabbè", "Dai su", "Ci vediamo dopo",
    "Ho appena letto", "Scusa il ritardo", "Tutto ok?", "Come sta la nonna?",
]

MEDIUM = [
    "Allora raga, io direi di vederci venerdì sera verso le nove, che dite?",
    "Ieri sera ho visto una puntata assurda di quella serie, ve la consiglio",
    "Ho prenotato il ristorante per sabato, mi hanno detto che si mangia bene",
    "Ragazzi mi sono perso la partita, chi mi racconta com'è andata?",
    "Sto organizzando il viaggio in Spagna, mando il programma quando è pronto",
    "Ma vi ricordate quella volta a Rimini che avevamo perso il treno?",
    "Devo assolutamente andare dal dentista, non ne posso più di questo dente",
    "Il lavoro procede bene, settimana prossima dovrei chiudere il progetto",
    "Ha iniziato a piovere forte, meglio prendere l'ombrello se uscite",
    "Ho trovato un posto fantastico per il trekking, ci andiamo in primavera?",
]

LONG = [
    "Allora ho parlato con il proprietario e mi ha detto che possiamo entrare dalle due, "
    "però bisogna portare i documenti e firmare, quindi portate la carta d'identità se no "
    "non ci fanno salire.",
    "Sono appena tornato dal viaggio, è stato incredibile: abbiamo visitato tre città in "
    "cinque giorni, mangiato cose assurde e conosciuto un sacco di gente. Vi racconto "
    "tutto con calma quando ci vediamo, ho mille foto da farvi vedere.",
    "Ragazzi vi aggiorno sulla situazione: il volo è stato spostato alle sette di mattina, "
    "quindi dobbiamo essere in aeroporto per le cinque, il che significa alzarsi alle tre "
    "e mezza. Se qualcuno vuole dormire da me la sera prima si organizzi.",
]

EVERYDAY = ["dai", "boh", "vabbè", "secondo me", "comunque", "ti giuro", "davvero", "esatto"]
NOUNS = ["pizza", "mare", "treno", "film", "concerto", "partita", "esame", "lavoro", "weekend",
         "birra", "montagna", "vacanza", "compleanno", "regalo", "macchina", "musica"]


def _text(rng: random.Random, long_bias: bool = False) -> str:
    roll = rng.random()
    if long_bias:
        roll = min(1.0, roll + 0.25)
    if roll < 0.62:
        base = rng.choice(SHORT)
        if rng.random() < 0.25:
            base = f"{base} {rng.choice(EVERYDAY)} {rng.choice(NOUNS)}"
    elif roll < 0.86:
        base = rng.choice(MEDIUM)
    elif roll < 0.985:
        base = rng.choice(LONG)
    else:
        base = f"{rng.choice(LONG)} {rng.choice(LONG)}"
    if rng.random() < 0.12:
        base += "\n" + rng.choice(["PS: niente panico 😂", "PPS: porta anche la roba", "Vi allego dopo"])
    if rng.random() < 0.10:
        base = f"{base} (questo messaggio è stato modificato)"
    return base


def _timestamps(rng: random.Random, start: datetime, end: datetime, count: int) -> list[datetime]:
    """Distribute ``count`` timestamps across [start, end] with seasonality."""
    days = (end.date() - start.date()).days + 1
    weights: list[float] = []
    for d in range(days):
        day = start + timedelta(days=d)
        w = 1.0
        if day.weekday() >= 5:
            w *= 1.45
        if day.month in (7, 8):
            w *= 1.3
        if day.month == 8:
            w *= 1.15
        w *= 1.0 + d / days * 0.5  # chats grow over the years
        weights.append(w)
    total = sum(weights)
    per_day = [max(0.0, weight / total * count) for weight in weights]

    out: list[datetime] = []
    for d, expected in enumerate(per_day):
        for _ in range(int(expected)):
            out.append(start + timedelta(days=d))
        frac = expected - int(expected)
        if rng.random() < frac:
            out.append(start + timedelta(days=d))
    while len(out) < count:
        out.append(start + timedelta(days=rng.randrange(days)))
    out = out[:count]

    # Give each message a realistic time of day, then sort within the whole set.
    hour_weights = [
        1, 1, 1, 1, 1, 1, 2, 3, 4, 5, 6, 7,
        7, 6, 5, 4, 5, 6, 8, 10, 12, 13, 10, 6,
    ]
    stamp = []
    for base in out:
        hour = rng.choices(range(24), weights=hour_weights)[0]
        minute = rng.randrange(60)
        second = rng.randrange(60)
        stamp.append(base.replace(hour=hour, minute=minute, second=second))
    stamp.sort()
    return stamp


def _line(dt: datetime, sender: str | None, text: str) -> str:
    stamp = dt.strftime("%d/%m/%Y, %H:%M")
    if sender is None:
        return f"{stamp} - {text}"
    return f"{stamp} - {sender}: {text}"


def generate_chat(
    rng: random.Random,
    name: str,
    senders: list[str],
    count: int,
    start: datetime,
    end: datetime,
    system_ratio: float = 0.012,
) -> list[str]:
    times = _timestamps(rng, start, end, count)
    lines: list[str] = []
    for i, dt in enumerate(times):
        roll = rng.random()
        if roll < system_ratio:
            lines.append(_line(dt, None, rng.choice(SYSTEM_LINES)))
            continue
        if rng.random() < 0.006:
            lines.append(_line(dt, None, rng.choice(CALL_LINES)))
            continue

        sender = rng.choice(senders)
        if rng.random() < 0.02:
            text = rng.choice(MEDIA_LINES)
        elif rng.random() < 0.01:
            text = rng.choice(VOICE_LINES)
        elif rng.random() < 0.004:
            text = "Questo messaggio è stato eliminato"
        else:
            text = _text(rng, long_bias=(i % 97 == 0))
        lines.append(_line(dt, sender, text))

        # Duplicate an identical message occasionally to exercise id suffixes.
        if rng.random() < 0.003:
            lines.append(_line(dt, sender, text))
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/exports"))
    parser.add_argument("--messages", type=int, default=150_000, help="Total messages (approx)")
    parser.add_argument("--private-messages", type=int, default=4_000)
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2026-07-01")
    parser.add_argument("--seed", type=int, default=20240915)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    start = datetime.fromisoformat(args.start)
    end = datetime.fromisoformat(args.end)

    group = generate_chat(rng, GROUP_NAME, GROUP_SENDERS, args.messages, start, end)
    private = generate_chat(
        rng, PRIVATE_NAME, PRIVATE_SENDERS, args.private_messages, start, end, system_ratio=0.0
    )

    args.out.mkdir(parents=True, exist_ok=True)
    for name, lines in ((GROUP_NAME, group), (PRIVATE_NAME, private)):
        path = args.out / f"{PREFIX}{name}.txt"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        size = path.stat().st_size / (1024 * 1024)
        print(f"wrote {path}  messages={len(lines):,}  size={size:.2f} MiB")


if __name__ == "__main__":
    main()
