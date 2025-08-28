"""
Test suite for the RAG Arbitration Detection System.

Contains unit tests, integration tests, and performance benchmarks.
"""

import pytest
from pathlib import Path

# Test configuration
TEST_DATA_DIR = Path(__file__).parent / "data"
FIXTURES_DIR = Path(__file__).parent / "fixtures"

__all__ = [
    "TEST_DATA_DIR",
    "FIXTURES_DIR"
]