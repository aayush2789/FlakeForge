"""Module-level cache dict that accumulates across tests."""

_cache = {}

def get_or_compute(key: str, compute_fn) -> any:
    """Memoize compute_fn result. Bug: _cache never cleared."""
    if key not in _cache:
        _cache[key] = compute_fn()
    return _cache[key]

def cache_size() -> int:
    return len(_cache)

def invalidate(key: str):
    _cache.pop(key, None)

def clear_cache():
    """Full reset — not called by tests."""
    _cache.clear()
