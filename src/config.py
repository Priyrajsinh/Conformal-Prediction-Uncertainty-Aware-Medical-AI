"""YAML config loader shared across the pipeline."""

from typing import Any

import yaml


def load_config(path: str) -> dict[str, Any]:
    """Load and return a YAML config file as a plain dict."""
    with open(path) as f:
        return dict(yaml.safe_load(f))
