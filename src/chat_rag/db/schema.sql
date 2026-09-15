PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS chats (
    chat_id       TEXT PRIMARY KEY,
    name          TEXT NOT NULL,
    is_group      INTEGER NOT NULL DEFAULT 0,
    source_file   TEXT,
    first_ts      INTEGER,
    last_ts       INTEGER,
    message_count INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS senders (
    sender_id   TEXT PRIMARY KEY,
    chat_id     TEXT NOT NULL REFERENCES chats(chat_id) ON DELETE CASCADE,
    name        TEXT NOT NULL,
    is_me       INTEGER,
    first_ts    INTEGER,
    last_ts     INTEGER,
    message_count INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS messages (
    id          TEXT PRIMARY KEY,
    chat_id     TEXT NOT NULL REFERENCES chats(chat_id) ON DELETE CASCADE,
    sender_id   TEXT REFERENCES senders(sender_id) ON DELETE SET NULL,
    ts          INTEGER NOT NULL,
    ts_local    TEXT NOT NULL,
    local_date  TEXT NOT NULL,
    local_time  TEXT NOT NULL,
    text        TEXT NOT NULL,
    raw_line    TEXT NOT NULL,
    msg_type    TEXT NOT NULL,
    reply_to_id TEXT,
    media_ref   TEXT,
    source_file TEXT,
    line_no     INTEGER,
    ingested_at INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_messages_chat_ts ON messages(chat_id, ts);
CREATE INDEX IF NOT EXISTS idx_messages_type ON messages(msg_type);
CREATE INDEX IF NOT EXISTS idx_messages_sender ON messages(sender_id);
CREATE INDEX IF NOT EXISTS idx_messages_date ON messages(local_date);

CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
    text,
    content='messages',
    content_rowid='rowid',
    tokenize='unicode61 remove_diacritics 2'
);

CREATE TRIGGER IF NOT EXISTS messages_ai AFTER INSERT ON messages BEGIN
    INSERT INTO messages_fts(rowid, text) VALUES (new.rowid, new.text);
END;

CREATE TRIGGER IF NOT EXISTS messages_ad AFTER DELETE ON messages BEGIN
    INSERT INTO messages_fts(messages_fts, rowid, text) VALUES ('delete', old.rowid, old.text);
END;

CREATE TRIGGER IF NOT EXISTS messages_au AFTER UPDATE ON messages BEGIN
    INSERT INTO messages_fts(messages_fts, rowid, text) VALUES ('delete', old.rowid, old.text);
    INSERT INTO messages_fts(rowid, text) VALUES (new.rowid, new.text);
END;

CREATE TABLE IF NOT EXISTS ingestion_runs (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    source_file  TEXT NOT NULL,
    file_hash    TEXT NOT NULL,
    chat_id      TEXT NOT NULL,
    started_at   INTEGER NOT NULL,
    finished_at  INTEGER,
    total_lines  INTEGER NOT NULL DEFAULT 0,
    parsed       INTEGER NOT NULL DEFAULT 0,
    inserted     INTEGER NOT NULL DEFAULT 0,
    skipped      INTEGER NOT NULL DEFAULT 0,
    continuations INTEGER NOT NULL DEFAULT 0,
    status       TEXT NOT NULL DEFAULT 'running'
);

CREATE INDEX IF NOT EXISTS idx_runs_file ON ingestion_runs(source_file, file_hash);
