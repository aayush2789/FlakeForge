"""Simulated DNS resolver with random latency."""
import random
import time

_dns_table = {
    "api.example.com": "10.0.0.1",
    "db.example.com": "10.0.0.2",
    "cache.example.com": "10.0.0.3",
}

def resolve(hostname: str, timeout: float = 0.1) -> str:
    """Resolve hostname with simulated latency.
    Bug: random delay can exceed timeout, causing resolution failure.
    """
    delay = random.uniform(0.01, 0.2)
    time.sleep(delay)
    if delay > timeout:
        raise TimeoutError(f"DNS resolution timed out for {hostname}")
    return _dns_table.get(hostname, "0.0.0.0")

def resolve_all(hostnames: list, timeout: float = 0.1) -> dict:
    results = {}
    for h in hostnames:
        try:
            results[h] = resolve(h, timeout)
        except TimeoutError:
            results[h] = None
    return results
