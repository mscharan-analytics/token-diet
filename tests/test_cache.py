import os
import tempfile

import pytest

from token_diet.cache import SQLiteCache


def test_sqlite_cache_lifecycle():
    # Use a temporary file for testing
    fd, temp_db_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)

    try:
        cache = SQLiteCache(temp_db_path)

        # Test empty fetch
        assert cache.get("ctx_nonexistent") is None

        # Test store and fetch
        cache.put("ctx_test_key", "This is the original raw content payload.")
        assert cache.get("ctx_test_key") == "This is the original raw content payload."

        # Test update (override)
        cache.put("ctx_test_key", "Updated original payload.")
        assert cache.get("ctx_test_key") == "Updated original payload."

        # Test clear
        cache.clear()
        assert cache.get("ctx_test_key") is None

    finally:
        if os.path.exists(temp_db_path):
            os.remove(temp_db_path)


def test_cache_fallback():
    # Attempt to initialize cache in a completely read-only or invalid system directory
    # (e.g. non-existent system directory under root that we don't have access to)
    invalid_path = "/nonexistent_system_dir_abc123/cache.db"

    # Should warn and gracefully activate fallback without crashing
    with pytest.warns(RuntimeWarning, match="falling back"):
        cache = SQLiteCache(invalid_path)

    # Verify put and get work fine inside the fallback in-memory dictionary
    cache.put("ctx_fallback_test", "Failsafe in-memory payload.")
    assert cache.get("ctx_fallback_test") == "Failsafe in-memory payload."
