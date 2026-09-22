import os
import sqlite3
import secrets
import numpy as np
from datetime import datetime, timezone
from contextlib import contextmanager
from werkzeug.security import generate_password_hash

SCHEMA = """
CREATE TABLE IF NOT EXISTS persons (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT UNIQUE NOT NULL,
    notes       TEXT,
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS embeddings (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id   INTEGER NOT NULL REFERENCES persons(id) ON DELETE CASCADE,
    vector      BLOB NOT NULL,
    dim         INTEGER NOT NULL,
    image_path  TEXT,
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS verification_logs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id       INTEGER REFERENCES persons(id) ON DELETE SET NULL,
    matched_name    TEXT,
    score           REAL,
    decision        TEXT NOT NULL,      -- matched | unknown | review | no_face | liveness_fail
    liveness_passed INTEGER,            -- 0/1/NULL (NULL = not checked)
    liveness_detail TEXT,
    image_path      TEXT,
    source          TEXT NOT NULL DEFAULT 'web',  -- web | api
    ip_address      TEXT,
    created_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS settings (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_embeddings_person ON embeddings(person_id);
CREATE INDEX IF NOT EXISTS idx_logs_created ON verification_logs(created_at);
"""


def now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class DB:
    """Thin synchronous SQLite wrapper. Flask runs the dev server single-process,
    so a simple per-request connection (via Flask's `g`) is enough here."""

    def __init__(self, path):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._init_schema()

    def _init_schema(self):
        with self.connect() as conn:
            conn.executescript(SCHEMA)
            conn.commit()

    @contextmanager
    def connect(self):
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
        finally:
            conn.close()

    # ---------------- settings ----------------
    def get_setting(self, key, default=None):
        with self.connect() as conn:
            row = conn.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
            return row["value"] if row else default

    def set_setting(self, key, value):
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO settings (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, str(value)),
            )
            conn.commit()

    def ensure_setting(self, key, default_value):
        current = self.get_setting(key)
        if current is None:
            self.set_setting(key, default_value)
            return str(default_value)
        return current

    # ---------------- persons ----------------
    def create_person(self, name, notes=None):
        with self.connect() as conn:
            cur = conn.execute(
                "INSERT INTO persons (name, notes, created_at) VALUES (?, ?, ?)",
                (name, notes, now_iso()),
            )
            conn.commit()
            return cur.lastrowid

    def get_person_by_name(self, name):
        with self.connect() as conn:
            return conn.execute("SELECT * FROM persons WHERE name = ?", (name,)).fetchone()

    def get_person(self, person_id):
        with self.connect() as conn:
            return conn.execute("SELECT * FROM persons WHERE id = ?", (person_id,)).fetchone()

    def list_persons(self):
        with self.connect() as conn:
            return conn.execute(
                """SELECT p.*, COUNT(e.id) AS embedding_count
                   FROM persons p LEFT JOIN embeddings e ON e.person_id = p.id
                   GROUP BY p.id ORDER BY p.name COLLATE NOCASE"""
            ).fetchall()

    def delete_person(self, person_id):
        with self.connect() as conn:
            conn.execute("DELETE FROM persons WHERE id = ?", (person_id,))
            conn.commit()

    # ---------------- embeddings ----------------
    def add_embedding(self, person_id, vector: np.ndarray, image_path=None):
        vector = np.asarray(vector, dtype=np.float32)
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO embeddings (person_id, vector, dim, image_path, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (person_id, vector.tobytes(), vector.shape[0], image_path, now_iso()),
            )
            conn.commit()

    def all_embeddings(self):
        """Returns (names, person_ids, vectors[N,D]) for every stored embedding."""
        with self.connect() as conn:
            rows = conn.execute(
                """SELECT e.vector, e.dim, p.id AS person_id, p.name AS name
                   FROM embeddings e JOIN persons p ON p.id = e.person_id"""
            ).fetchall()
        names, ids, vecs = [], [], []
        for r in rows:
            vecs.append(np.frombuffer(r["vector"], dtype=np.float32))
            names.append(r["name"])
            ids.append(r["person_id"])
        return names, ids, vecs

    def delete_embedding(self, embedding_id):
        with self.connect() as conn:
            conn.execute("DELETE FROM embeddings WHERE id = ?", (embedding_id,))
            conn.commit()

    def embeddings_for_person(self, person_id):
        with self.connect() as conn:
            return conn.execute(
                "SELECT * FROM embeddings WHERE person_id = ? ORDER BY created_at DESC", (person_id,)
            ).fetchall()

    # ---------------- verification logs ----------------
    def add_log(self, *, person_id, matched_name, score, decision, liveness_passed,
                liveness_detail, image_path, source, ip_address):
        with self.connect() as conn:
            conn.execute(
                """INSERT INTO verification_logs
                   (person_id, matched_name, score, decision, liveness_passed, liveness_detail,
                    image_path, source, ip_address, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (person_id, matched_name, score, decision,
                 None if liveness_passed is None else int(liveness_passed),
                 liveness_detail, image_path, source, ip_address, now_iso()),
            )
            conn.commit()

    def recent_logs(self, limit=100, person_id=None):
        with self.connect() as conn:
            if person_id:
                return conn.execute(
                    "SELECT * FROM verification_logs WHERE person_id = ? "
                    "ORDER BY created_at DESC LIMIT ?", (person_id, limit)
                ).fetchall()
            return conn.execute(
                "SELECT * FROM verification_logs ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()

    def stats(self):
        with self.connect() as conn:
            total = conn.execute("SELECT COUNT(*) c FROM verification_logs").fetchone()["c"]
            matched = conn.execute(
                "SELECT COUNT(*) c FROM verification_logs WHERE decision = 'matched'"
            ).fetchone()["c"]
            unknown = conn.execute(
                "SELECT COUNT(*) c FROM verification_logs WHERE decision = 'unknown'"
            ).fetchone()["c"]
            liveness_fail = conn.execute(
                "SELECT COUNT(*) c FROM verification_logs WHERE decision = 'liveness_fail'"
            ).fetchone()["c"]
            people = conn.execute("SELECT COUNT(*) c FROM persons").fetchone()["c"]
            return {
                "total_attempts": total,
                "matched": matched,
                "unknown": unknown,
                "liveness_fail": liveness_fail,
                "people": people,
            }


def bootstrap_defaults(db: DB, cfg):
    """Seed settings table with defaults and generate admin/API credentials if missing."""
    db.ensure_setting("match_threshold", cfg.MATCH_THRESHOLD_DEFAULT)
    db.ensure_setting("review_margin", cfg.REVIEW_MARGIN)
    db.ensure_setting("require_liveness", "1" if cfg.REQUIRE_LIVENESS_DEFAULT else "0")

    if db.get_setting("admin_password_hash") is None:
        password = cfg.ADMIN_PASSWORD or secrets.token_urlsafe(9)
        db.set_setting("admin_password_hash", generate_password_hash(password))
        db.set_setting("admin_username", cfg.ADMIN_USERNAME)
        if not cfg.ADMIN_PASSWORD:
            print("=" * 60)
            print(f" Generated admin login  ->  user: {cfg.ADMIN_USERNAME}  password: {password}")
            print(" Set ADMIN_PASSWORD in your environment to override this.")
            print("=" * 60)

    if db.get_setting("api_key") is None:
        api_key = cfg.API_KEY or secrets.token_urlsafe(24)
        db.set_setting("api_key", api_key)
        if not cfg.API_KEY:
            print(f" Generated API key -> {api_key}  (set API_KEY env var to override)")
