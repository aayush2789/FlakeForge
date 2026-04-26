"""Token generator that uses random.choice without seeding."""
import random
import string

def generate_token(length: int = 8) -> str:
    """Generate a random alphanumeric token. Bug: no seed, not deterministic."""
    chars = string.ascii_lowercase + string.digits
    return "".join(random.choice(chars) for _ in range(length))

def generate_pair() -> tuple:
    """Generate two tokens. They will almost never match."""
    return generate_token(), generate_token()
