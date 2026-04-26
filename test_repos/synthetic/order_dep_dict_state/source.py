"""Key-value store backed by a module-level dict."""

_store = {}

def put(key: str, value):
    _store[key] = value

def get(key: str):
    return _store.get(key)

def has(key: str) -> bool:
    return key in _store

def delete(key: str):
    _store.pop(key, None)

def clear():
    _store.clear()
