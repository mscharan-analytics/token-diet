import os
import sqlite3
from pathlib import Path
from typing import Optional

class SQLiteCache:
    """
    A lightweight SQLite-backed persistent cache for storing the original,
    uncompressed content payloads.
    """
    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            # Default to ~/.config/token_diet/cache.db
            home = Path.home()
            config_dir = home / ".config" / "token_diet"
            try:
                config_dir.mkdir(parents=True, exist_ok=True)
                self.db_path = str(config_dir / "cache.db")
            except OSError:
                # Fallback to local working directory if home dir is not writeable
                self.db_path = "token_diet_cache.db"
        else:
            self.db_path = db_path
            
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        """Create a database connection."""
        conn = sqlite3.connect(self.db_path)
        # Return rows as dict-like objects if needed, but simple is better.
        return conn

    def _init_db(self) -> None:
        """Create the table if it does not exist."""
        with self._get_conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS context_cache (
                    id TEXT PRIMARY KEY,
                    original_text TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()

    def put(self, cache_id: str, original_text: str) -> None:
        """Store the original uncompressed text in the cache."""
        with self._get_conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO context_cache (id, original_text) VALUES (?, ?)",
                (cache_id, original_text)
            )
            conn.commit()

    def get(self, cache_id: str) -> Optional[str]:
        """Retrieve the original uncompressed text from the cache."""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT original_text FROM context_cache WHERE id = ?", (cache_id,))
            row = cursor.fetchone()
            return row[0] if row else None

    def clear(self) -> None:
        """Clear all cached entries."""
        with self._get_conn() as conn:
            conn.execute("DELETE FROM context_cache")
            conn.commit()
