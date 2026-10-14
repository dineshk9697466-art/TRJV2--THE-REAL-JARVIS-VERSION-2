
import sqlite3
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data" / "trjv2.db"


def get_connection():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    with get_connection() as conn:
        # Chat sessions
        conn.execute("""
            CREATE TABLE IF NOT EXISTS chats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL DEFAULT 'New Chat',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Messages: preserve the existing table and its data
        conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Add chat_id to an existing messages table when needed
        columns = {
            row["name"]
            for row in conn.execute(
                "PRAGMA table_info(messages)"
            ).fetchall()
        }

        if "chat_id" not in columns:
            conn.execute("""
                ALTER TABLE messages
                ADD COLUMN chat_id INTEGER
                REFERENCES chats(id) ON DELETE CASCADE
            """)

        # Keep existing messages by grouping them into a legacy chat
        old_messages = conn.execute("""
            SELECT COUNT(*) AS total
            FROM messages
            WHERE chat_id IS NULL
        """).fetchone()["total"]

        if old_messages:
            cursor = conn.execute("""
                INSERT INTO chats (title)
                VALUES ('Previous Chat History')
            """)
            legacy_chat_id = cursor.lastrowid

            conn.execute("""
                UPDATE messages
                SET chat_id = ?
                WHERE chat_id IS NULL
            """, (legacy_chat_id,))

        # Preserve the separate long-term memories feature
        conn.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                memory_key TEXT NOT NULL UNIQUE,
                memory_value TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)


def create_chat(title="New Chat"):
    with get_connection() as conn:
        cursor = conn.execute("""
            INSERT INTO chats (title)
            VALUES (?)
        """, (title.strip() or "New Chat",))

        return cursor.lastrowid


def get_chats():
    with get_connection() as conn:
        rows = conn.execute("""
            SELECT id, title, created_at, updated_at
            FROM chats
            ORDER BY updated_at DESC, id DESC
        """).fetchall()

    return [dict(row) for row in rows]


def get_chat(chat_id):
    with get_connection() as conn:
        row = conn.execute("""
            SELECT id, title, created_at, updated_at
            FROM chats
            WHERE id = ?
        """, (chat_id,)).fetchone()

    return dict(row) if row else None


def update_chat_title(chat_id, title):
    title = title.strip()

    if not title:
        return

    with get_connection() as conn:
        conn.execute("""
            UPDATE chats
            SET title = ?, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
        """, (title, chat_id))


def save_message(*args):
    # Backward-compatible format: save_message(role, content)
    if len(args) == 2:
        role, content = args
        chat_id = None

    # New format: save_message(chat_id, role, content)
    elif len(args) == 3:
        chat_id, role, content = args

    else:
        raise TypeError(
            "Use save_message(role, content) or "
            "save_message(chat_id, role, content)"
        )

    if role not in ("user", "assistant"):
        raise ValueError("Invalid role")

    with get_connection() as conn:
        if chat_id is not None:
            exists = conn.execute("""
                SELECT id FROM chats WHERE id = ?
            """, (chat_id,)).fetchone()

            if exists is None:
                raise ValueError(f"Chat {chat_id} does not exist")

        conn.execute("""
            INSERT INTO messages (chat_id, role, content)
            VALUES (?, ?, ?)
        """, (chat_id, role, content))

        if chat_id is not None:
            conn.execute("""
                UPDATE chats
                SET updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (chat_id,))


def get_chat_messages(chat_id, limit=None):
    with get_connection() as conn:
        if limit is None:
            rows = conn.execute("""
                SELECT id, chat_id, role, content, created_at
                FROM messages
                WHERE chat_id = ?
                ORDER BY id ASC
            """, (chat_id,)).fetchall()
        else:
            rows = conn.execute("""
                SELECT id, chat_id, role, content, created_at
                FROM (
                    SELECT id, chat_id, role, content, created_at
                    FROM messages
                    WHERE chat_id = ?
                    ORDER BY id DESC
                    LIMIT ?
                )
                ORDER BY id ASC
            """, (chat_id, limit)).fetchall()

    return [dict(row) for row in rows]


# Keep the old global history function working for the current app
def get_messages(limit=100):
    with get_connection() as conn:
        rows = conn.execute("""
            SELECT id, role, content, created_at
            FROM messages
            ORDER BY id DESC
            LIMIT ?
        """, (limit,)).fetchall()

    return [dict(row) for row in reversed(rows)]


def delete_chat(chat_id):
    with get_connection() as conn:
        conn.execute("""
            DELETE FROM chats
            WHERE id = ?
        """, (chat_id,))


def save_memory(memory_key, memory_value):
    with get_connection() as conn:
        conn.execute("""
            INSERT INTO memories (memory_key, memory_value)
            VALUES (?, ?)
            ON CONFLICT(memory_key) DO UPDATE SET
                memory_value = excluded.memory_value,
                updated_at = CURRENT_TIMESTAMP
        """, (memory_key, memory_value))


def get_memories():
    with get_connection() as conn:
        rows = conn.execute("""
            SELECT id, memory_key, memory_value
            FROM memories
            ORDER BY id
        """).fetchall()

    return [dict(row) for row in rows]


def delete_memory(memory_id):
    with get_connection() as conn:
        conn.execute("""
            DELETE FROM memories
            WHERE id = ?
        """, (memory_id,))
