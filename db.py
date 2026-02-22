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


def _safe_add_column(conn: sqlite3.Connection, table: str, column: str, coltype: str):
    """Add a column to a table, ignoring if it already exists."""
    try:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {coltype}")
    except sqlite3.OperationalError:
        pass


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
        CREATE TABLE IF NOT EXISTS model_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trained_at TIMESTAMP,
            num_training_samples INTEGER,
            accuracy FLOAT,
            precision_score FLOAT,
            recall_score FLOAT,
            f1_score FLOAT,
            model_path TEXT,
            confusion_matrix TEXT
        );
        CREATE TABLE IF NOT EXISTS full_predictions (
            contribution_id TEXT PRIMARY KEY REFERENCES contributions(id),
            prediction INTEGER,
            confidence FLOAT,
            predicted_at TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT
        );
    """)
    # Add model-feedback columns to labels (safe if they already exist)
    _safe_add_column(conn, "labels", "model_prediction", "INTEGER")
    _safe_add_column(conn, "labels", "model_confidence", "FLOAT")
    _safe_add_column(conn, "labels", "model_agreed", "INTEGER")
    conn.commit()
    conn.close()


def insert_contributions(rows: list[dict]) -> int:
    """Insert or update contributions. Returns number of rows affected.

    On conflict (duplicate id), refreshes matched_keywords and is_candidate
    so that keyword-list changes are reflected on existing rows.
    """
    conn = get_conn()
    for r in rows:
        conn.execute(
            """INSERT INTO contributions
               (id, text, word_count, speaker, house, debate_title, debate_date, matched_keywords, is_candidate)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(id) DO UPDATE SET
                   matched_keywords = excluded.matched_keywords,
                   is_candidate = excluded.is_candidate""",
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
    conn.commit()
    conn.close()
    return len(rows)


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


def purge_unlabelled_negatives() -> int:
    """Remove all unlabelled negative candidates. Labelled negatives are kept.

    Called before each ingestion so the negative pool is replaced
    rather than accumulating across runs.
    """
    conn = get_conn()
    cursor = conn.execute(
        """DELETE FROM contributions
           WHERE is_candidate = 0
             AND id NOT IN (SELECT contribution_id FROM labels)"""
    )
    deleted = cursor.rowcount
    conn.commit()
    conn.close()
    return deleted


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


def save_label(contribution_id: str, label: int, notes: str = "",
               model_prediction: int | None = None,
               model_confidence: float | None = None,
               model_agreed: int | None = None):
    conn = get_conn()
    conn.execute(
        """INSERT INTO labels
           (contribution_id, label, labelled_at, notes, model_prediction, model_confidence, model_agreed)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (contribution_id, label, datetime.now().isoformat(), notes,
         model_prediction, model_confidence, model_agreed),
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
                  c.is_candidate, c.word_count, c.matched_keywords, l.label
           FROM labels l
           JOIN contributions c ON l.contribution_id = c.id
           ORDER BY l.labelled_at"""
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_unlabelled_contributions() -> list[dict]:
    """Return all unlabelled contributions with fields needed for prediction."""
    conn = get_conn()
    rows = conn.execute(
        """SELECT c.id, c.text, c.word_count, c.speaker, c.house,
                  c.debate_title, c.debate_date, c.matched_keywords, c.is_candidate
           FROM contributions c
           LEFT JOIN labels l ON c.id = l.contribution_id
           WHERE l.id IS NULL
           ORDER BY c.debate_date, c.id"""
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ── Settings ───────────────────────────────────────────────────────────────────

def save_keywords(keywords: list[str]):
    """Persist the keyword list to the database."""
    conn = get_conn()
    conn.execute(
        "INSERT INTO settings (key, value) VALUES ('keywords', ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (json.dumps(keywords),),
    )
    conn.commit()
    conn.close()


def get_keywords() -> list[str] | None:
    """Load the persisted keyword list, or None if never saved."""
    conn = get_conn()
    row = conn.execute(
        "SELECT value FROM settings WHERE key = 'keywords'"
    ).fetchone()
    conn.close()
    if row is None:
        return None
    return json.loads(row["value"])


# ── Model-related queries ─────────────────────────────────────────────────────

def log_model_run(num_samples: int, accuracy: float, precision: float,
                  recall: float, f1: float, model_path: str,
                  confusion_matrix_data: list | None = None) -> int:
    conn = get_conn()
    cm_json = json.dumps(confusion_matrix_data) if confusion_matrix_data else None
    cursor = conn.execute(
        """INSERT INTO model_runs
           (trained_at, num_training_samples, accuracy, precision_score,
            recall_score, f1_score, model_path, confusion_matrix)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (datetime.now().isoformat(), num_samples, accuracy, precision,
         recall, f1, model_path, cm_json),
    )
    run_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return run_id


def get_model_runs() -> list[dict]:
    conn = get_conn()
    rows = conn.execute("SELECT * FROM model_runs ORDER BY id").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_latest_model_run() -> dict | None:
    conn = get_conn()
    row = conn.execute("SELECT * FROM model_runs ORDER BY id DESC LIMIT 1").fetchone()
    conn.close()
    return dict(row) if row else None


def get_rolling_agreement(n: int = 50) -> float | None:
    """Agreement rate over the last n labels that had a model prediction."""
    conn = get_conn()
    rows = conn.execute(
        """SELECT model_agreed FROM labels
           WHERE model_agreed IS NOT NULL
           ORDER BY labelled_at DESC LIMIT ?""",
        (n,),
    ).fetchall()
    conn.close()
    if not rows:
        return None
    return sum(r["model_agreed"] for r in rows) / len(rows)


def get_recent_disagreements(limit: int = 20) -> list[dict]:
    conn = get_conn()
    rows = conn.execute(
        """SELECT c.text, c.speaker, c.debate_title, c.debate_date,
                  l.label, l.model_prediction, l.model_confidence, l.labelled_at
           FROM labels l
           JOIN contributions c ON l.contribution_id = c.id
           WHERE l.model_agreed = 0
           ORDER BY l.labelled_at DESC LIMIT ?""",
        (limit,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def save_full_predictions(predictions: list[dict]):
    """Replace all full_predictions with a new batch."""
    conn = get_conn()
    conn.execute("DELETE FROM full_predictions")
    now = datetime.now().isoformat()
    conn.executemany(
        """INSERT INTO full_predictions (contribution_id, prediction, confidence, predicted_at)
           VALUES (?, ?, ?, ?)""",
        [(p["id"], p["prediction"], p["confidence"], now) for p in predictions],
    )
    conn.commit()
    conn.close()


def get_full_predictions_summary() -> list[dict]:
    """Prediction counts grouped by house and prediction label."""
    conn = get_conn()
    rows = conn.execute(
        """SELECT c.house, fp.prediction, COUNT(*) as count
           FROM full_predictions fp
           JOIN contributions c ON fp.contribution_id = c.id
           GROUP BY c.house, fp.prediction
           ORDER BY c.house, fp.prediction"""
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# Always ensure tables exist on import
init_db()
