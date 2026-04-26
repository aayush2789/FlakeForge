"""Tests that assume deterministic random output.

test_token_is_alpha_only is FLAKY because tokens contain digits too,
and test_pair_first_starts_with_a is FLAKY because it assumes the first
char is 'a'.
"""
from source import generate_token, generate_pair

def test_token_is_alpha_only():
    """FLAKY — token contains letters AND digits, so isalpha() fails ~75% of time."""
    token = generate_token()
    assert token.isalpha()

def test_token_length():
    assert len(generate_token(10)) == 10

def test_pair_different():
    a, b = generate_pair()
    assert a != b
