"""User lookup service with LRU cache."""
from functools import lru_cache

_user_db = {
    1: "Alice",
    2: "Bob",
    3: "Charlie",
}

@lru_cache(maxsize=32)
def get_user(user_id: int) -> str:
    """Lookup user by ID. Bug: lru_cache not cleared between tests."""
    return _user_db.get(user_id, "Unknown")

def add_user(user_id: int, name: str):
    _user_db[user_id] = name

def remove_user(user_id: int):
    _user_db.pop(user_id, None)

def reset_db():
    global _user_db
    _user_db = {1: "Alice", 2: "Bob", 3: "Charlie"}
    get_user.cache_clear()
