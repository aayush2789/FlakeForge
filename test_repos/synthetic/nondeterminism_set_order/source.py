"""Tag deduplication using sets — iteration order is not guaranteed."""

def unique_tags(tags: list) -> list:
    """Return deduplicated tags. Bug: converts to set then back to list,
    so output order is non-deterministic."""
    return list(set(tags))

def first_tag(tags: list) -> str:
    """Return the 'first' unique tag. Non-deterministic due to set ordering."""
    uniq = unique_tags(tags)
    return uniq[0] if uniq else ""
