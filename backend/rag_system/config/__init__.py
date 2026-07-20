"""
Configuration files and settings.

Contains environment-specific configurations, model parameters,
and system settings.
"""

from pathlib import Path

CONFIG_DIR = Path(__file__).parent

__all__ = [
    "CONFIG_DIR"
]