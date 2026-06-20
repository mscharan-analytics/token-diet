import sqlite3
import warnings
from pathlib import Path


class SQLiteCache:
    """
    A lightweight SQLite-backed persistent cache for storing the original,
    uncompressed content payloads.

    Failsafe: If database creation or reads/writes fail (e.g., read-only filesystem,
    permission errors, disk space exhaustion), it gracefully falls back to an
    in-memory dictionary cache to ensure production pipelines never crash.
    """

    def __init__(self, db_path: str | None = None):
        self.db_path: str | None = None
        self._use_fallback = False
        self._mem_cache: dict[str, str] = {}

        if db_path is None:
            # Default to ~/.config/token_diet/cache.db
            try:
                home = Path.home()
                config_dir = home / ".config" / "token_diet"
                config_dir.mkdir(parents=True, exist_ok=True)
                self.db_path = str(config_dir / "cache.db")
            except Exception as e:
                # If we cannot parse/write to home folder, try local working directory
                warnings.warn(
                    f"Failed to access home config folder, trying local DB: {e}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self.db_path = "token_diet_cache.db"
        else:
            self.db_path = db_path

        try:
            self._init_db()
        except Exception as e:
            warnings.warn(
                f"Failed to initialize SQLite database cache: {e}. Token Diet is falling back to an in-memory cache.",
                RuntimeWarning, stacklevel=2,
            )
            self._use_fallback = True

    def _get_conn(self) -> sqlite3.Connection:
        """Create a database connection."""
        if self.db_path is None:
            raise sqlite3.OperationalError("No database path configured.")
        return sqlite3.connect(self.db_path)

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
        if self._use_fallback:
            self._mem_cache[cache_id] = original_text
            return

        try:
            with self._get_conn() as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO context_cache (id, original_text) VALUES (?, ?)", (cache_id, original_text)
                )
                conn.commit()
        except Exception as e:
            warnings.warn(f"Failed to write to SQLite cache: {e}. Storing in memory.", RuntimeWarning, stacklevel=2)
            # Failsafe: Write to in-memory cache as fallback
            self._mem_cache[cache_id] = original_text

    def get(self, cache_id: str) -> str | None:
        """Retrieve the original uncompressed text from the cache."""
        # Always check in-memory fallback first (in case some keys were stored there during runtime fallback)
        if cache_id in self._mem_cache:
            return self._mem_cache[cache_id]

        if self._use_fallback:
            return None

        try:
            with self._get_conn() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT original_text FROM context_cache WHERE id = ?", (cache_id,))
                row = cursor.fetchone()
                return row[0] if row else None
        except Exception as e:
            warnings.warn(f"Failed to read from SQLite cache: {e}.", RuntimeWarning, stacklevel=2)
            return None

    def clear(self) -> None:
        """Clear all cached entries."""
        self._mem_cache.clear()
        if self._use_fallback:
            return

        try:
            with self._get_conn() as conn:
                conn.execute("DELETE FROM context_cache")
                conn.commit()
        except Exception as e:
            warnings.warn(f"Failed to clear SQLite cache: {e}.", RuntimeWarning, stacklevel=2)
