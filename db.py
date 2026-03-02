import sqlite3
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "history.db")

def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_conn()
    cur = conn.cursor()

    # Users table (admin supported)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            is_admin INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL
        )
    """)

    # Checks table (linked to user)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS checks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            amount REAL NOT NULL,
            hour INTEGER NOT NULL,
            new_device INTEGER NOT NULL,
            transaction_count INTEGER NOT NULL,
            fraud_prob REAL NOT NULL,
            label INTEGER NOT NULL,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    """)

    # Model metadata
    cur.execute("""
        CREATE TABLE IF NOT EXISTS model_meta (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            accuracy REAL,
            label_col TEXT,
            notes TEXT
        )
    """)

    conn.commit()
    conn.close()

# ---------- Users ----------
def create_user(username, password_hash, created_at, is_admin=0):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO users (username, password_hash, is_admin, created_at) VALUES (?, ?, ?, ?)",
        (username, password_hash, is_admin, created_at)
    )
    conn.commit()
    conn.close()

def get_user_by_username(username):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE username=?", (username,))
    row = cur.fetchone()
    conn.close()
    return row

def get_user_by_id(user_id):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE id=?", (user_id,))
    row = cur.fetchone()
    conn.close()
    return row

def list_users():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT id, username, is_admin, created_at FROM users ORDER BY id DESC")
    rows = cur.fetchall()
    conn.close()
    return rows

# ---------- Checks ----------
def add_check(user_id, created_at, amount, hour, new_device, transaction_count, fraud_prob, label):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO checks (user_id, created_at, amount, hour, new_device, transaction_count, fraud_prob, label)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (user_id, created_at, amount, hour, new_device, transaction_count, fraud_prob, label))
    conn.commit()
    conn.close()

def get_recent_checks(user_id=None, limit=20):
    conn = get_conn()
    cur = conn.cursor()
    if user_id is None:
        cur.execute("""
            SELECT c.*, u.username FROM checks c
            JOIN users u ON u.id = c.user_id
            ORDER BY c.id DESC
            LIMIT ?
        """, (limit,))
    else:
        cur.execute("""
            SELECT c.*, u.username FROM checks c
            JOIN users u ON u.id = c.user_id
            WHERE c.user_id = ?
            ORDER BY c.id DESC
            LIMIT ?
        """, (user_id, limit))
    rows = cur.fetchall()
    conn.close()
    return rows

def clear_checks(user_id=None):
    conn = get_conn()
    cur = conn.cursor()
    if user_id is None:
        cur.execute("DELETE FROM checks")
    else:
        cur.execute("DELETE FROM checks WHERE user_id=?", (user_id,))
    conn.commit()
    conn.close()

# ---------- Analytics ----------
def analytics_summary(user_id=None):
    conn = get_conn()
    cur = conn.cursor()

    if user_id is None:
        cur.execute("SELECT COUNT(*) AS total, COALESCE(SUM(label),0) AS fraud FROM checks")
        row = cur.fetchone()
        total = int(row["total"])
        fraud = int(row["fraud"])
        safe = total - fraud

        cur.execute("""
            SELECT hour, COUNT(*) AS n
            FROM checks
            WHERE label=1
            GROUP BY hour
            ORDER BY hour
        """)
        by_hour = [{"hour": int(r["hour"]), "n": int(r["n"])} for r in cur.fetchall()]

        cur.execute("""
            SELECT new_device, COUNT(*) AS n
            FROM checks
            WHERE label=1
            GROUP BY new_device
        """)
        by_device = [{"new_device": int(r["new_device"]), "n": int(r["n"])} for r in cur.fetchall()]
    else:
        cur.execute("SELECT COUNT(*) AS total, COALESCE(SUM(label),0) AS fraud FROM checks WHERE user_id=?", (user_id,))
        row = cur.fetchone()
        total = int(row["total"])
        fraud = int(row["fraud"])
        safe = total - fraud

        cur.execute("""
            SELECT hour, COUNT(*) AS n
            FROM checks
            WHERE user_id=? AND label=1
            GROUP BY hour
            ORDER BY hour
        """, (user_id,))
        by_hour = [{"hour": int(r["hour"]), "n": int(r["n"])} for r in cur.fetchall()]

        cur.execute("""
            SELECT new_device, COUNT(*) AS n
            FROM checks
            WHERE user_id=? AND label=1
            GROUP BY new_device
        """, (user_id,))
        by_device = [{"new_device": int(r["new_device"]), "n": int(r["n"])} for r in cur.fetchall()]

    conn.close()
    return {"total": total, "fraud": fraud, "safe": safe, "by_hour": by_hour, "by_device": by_device}

# ---------- Model meta ----------
def save_model_meta(created_at, accuracy=None, label_col=None, notes=None):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO model_meta (created_at, accuracy, label_col, notes) VALUES (?, ?, ?, ?)",
        (created_at, accuracy, label_col, notes)
    )
    conn.commit()
    conn.close()

def latest_model_meta():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM model_meta ORDER BY id DESC LIMIT 1")
    row = cur.fetchone()
    conn.close()
    return row