"""SQLite database setup and operations for the labelling app."""

import json
import sqlite3
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).parent / "labelling.db"


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS contributions (
            id TEXT PRIMARY KEY,
            text TEXT,
            word_count INTEGER,
            speaker TEXT,
            house TEXT,
            debate_title TEXT,
            debate_date TEXT,
            matched_keywords TEXT,
            is_candidate INTEGER
        );
        CREATE TABLE IF NOT EXISTS labels (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            contribution_id TEXT REFERENCES contributions(id),
            label INTEGER,
            labelled_at TIMESTAMP,
            notes TEXT
        );
    """)
    conn.commit()
    conn.close()


def insert_contributions(rows: list[dict]) -> int:
    """Insert contributions, skipping duplicates. Returns count of newly inserted rows."""
    conn = get_conn()
    inserted = 0
    for r in rows:
        try:
            conn.execute(
                """INSERT OR IGNORE INTO contributions
                   (id, text, word_count, speaker, house, debate_title, debate_date, matched_keywords, is_candidate)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    r["id"],
                    r["text"],
                    r["word_count"],
                    r["speaker"],
                    r["house"],
                    r["debate_title"],
                    r["debate_date"],
                    json.dumps(r.get("matched_keywords", [])),
                    r["is_candidate"],
                ),
            )
            if conn.total_changes > inserted:
                inserted = conn.total_changes
        except sqlite3.IntegrityError:
            pass
    count = conn.total_changes
    conn.commit()
    conn.close()
    return count


def purge_stale_unlabelled(fresh_positive_ids: set[str]) -> int:
    """Remove unlabelled positive candidates whose IDs are not in the fresh set.

    Labelled contributions are always kept, regardless of keyword changes.
    Negative samples are also kept (they don't depend on keywords).
    Returns the number of rows deleted.
    """
    conn = get_conn()
    # Find unlabelled positive candidates not in the new positive set
    existing = conn.execute(
        """SELECT c.id FROM contributions c
           LEFT JOIN labels l ON c.id = l.contribution_id
           WHERE l.id IS NULL AND c.is_candidate = 1"""
    ).fetchall()
    to_delete = [r["id"] for r in existing if r["id"] not in fresh_positive_ids]
    if to_delete:
        conn.executemany("DELETE FROM contributions WHERE id = ?", [(i,) for i in to_delete])
    conn.commit()
    conn.close()
    return len(to_delete)


def get_unlabelled_queue() -> list[dict]:
    """Return unlabelled contributions interleaved: positive, negative, positive, …"""
    conn = get_conn()
    positives = [
        dict(r)
        for r in conn.execute(
            """SELECT c.* FROM contributions c
               LEFT JOIN labels l ON c.id = l.contribution_id
               WHERE l.id IS NULL AND c.is_candidate = 1
               ORDER BY c.debate_date, c.id"""
        ).fetchall()
    ]
    negatives = [
        dict(r)
        for r in conn.execute(
            """SELECT c.* FROM contributions c
               LEFT JOIN labels l ON c.id = l.contribution_id
               WHERE l.id IS NULL AND c.is_candidate = 0
               ORDER BY RANDOM()"""
        ).fetchall()
    ]
    conn.close()

    # Interleave: one positive, one negative, repeat; append leftovers
    merged = []
    pi, ni = 0, 0
    while pi < len(positives) or ni < len(negatives):
        if pi < len(positives):
            merged.append(positives[pi])
            pi += 1
        if ni < len(negatives):
            merged.append(negatives[ni])
            ni += 1
    return merged


def save_label(contribution_id: str, label: int, notes: str = ""):
    conn = get_conn()
    conn.execute(
        "INSERT INTO labels (contribution_id, label, labelled_at, notes) VALUES (?, ?, ?, ?)",
        (contribution_id, label, datetime.now().isoformat(), notes),
    )
    conn.commit()
    conn.close()


def get_counts() -> dict:
    conn = get_conn()
    total = conn.execute("SELECT COUNT(*) FROM contributions").fetchone()[0]
    labelled = conn.execute("SELECT COUNT(*) FROM labels").fetchone()[0]
    relevant = conn.execute("SELECT COUNT(*) FROM labels WHERE label = 1").fetchone()[0]
    not_relevant = conn.execute("SELECT COUNT(*) FROM labels WHERE label = 0").fetchone()[0]
    positives = conn.execute("SELECT COUNT(*) FROM contributions WHERE is_candidate = 1").fetchone()[0]
    negatives = conn.execute("SELECT COUNT(*) FROM contributions WHERE is_candidate = 0").fetchone()[0]
    conn.close()
    return {
        "total": total,
        "labelled": labelled,
        "relevant": relevant,
        "not_relevant": not_relevant,
        "positives": positives,
        "negatives": negatives,
    }


def get_recent_labels(limit: int = 50) -> list[dict]:
    conn = get_conn()
    rows = conn.execute(
        """SELECT l.contribution_id, c.speaker, c.debate_title, c.debate_date,
                  l.label, l.notes, l.labelled_at
           FROM labels l
           JOIN contributions c ON l.contribution_id = c.id
           ORDER BY l.labelled_at DESC
           LIMIT ?""",
        (limit,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def export_labelled() -> list[dict]:
    conn = get_conn()
    rows = conn.execute(
        """SELECT c.id, c.text, c.speaker, c.house, c.debate_title, c.debate_date,
                  c.is_candidate, l.label
           FROM labels l
           JOIN contributions c ON l.contribution_id = c.id
           ORDER BY l.labelled_at"""
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# Always ensure tables exist on import
init_db()
