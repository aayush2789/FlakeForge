"""Tests for async pipeline.

test_pipeline_all_succeed is FLAKY because stage_two has 30% random
failure rate and pipeline() doesn't handle exceptions.
"""
import asyncio
from source import pipeline, safe_pipeline

def test_pipeline_all_succeed():
    """FLAKY — 30% chance each item fails in stage_two."""
    items = ["a", "b", "c"]
    results = asyncio.run(pipeline(items))
    assert len(results) == 3

def test_safe_pipeline():
    items = ["x", "y", "z"]
    results = asyncio.run(safe_pipeline(items))
    assert len(results) <= 3

def test_single_item():
    """Less flaky — only one item, 30% failure rate."""
    result = asyncio.run(pipeline(["solo"]))
    assert len(result) == 1
