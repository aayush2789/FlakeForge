"""String sorting utilities that depend on locale."""
import locale

def sorted_names(names: list) -> list:
    """Sort names using locale-aware comparison.
    Bug: locale.strxfrm behavior varies by platform and locale setting.
    """
    return sorted(names, key=locale.strxfrm)

def compare_names(a: str, b: str) -> int:
    """Compare two names locale-aware. Returns -1, 0, or 1."""
    result = locale.strcoll(a, b)
    if result < 0:
        return -1
    elif result > 0:
        return 1
    return 0

def normalize_for_sort(name: str) -> str:
    """Normalize for sorting: lowercase, strip accents (simplistic)."""
    return name.lower().strip()
