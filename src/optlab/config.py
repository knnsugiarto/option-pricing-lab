"""Load the TOML configuration that drives every study."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

REQUIRED_SECTIONS = ("contract", "monte_carlo", "crank_nicolson", "gamma_profile", "accuracy", "market")


def load_config(path: Path) -> dict[str, Any]:
    """Read a config file and check that every section is present."""
    with path.open("rb") as handle:
        config = tomllib.load(handle)
    missing = [section for section in REQUIRED_SECTIONS if section not in config]
    if missing or "seed" not in config:
        raise ValueError(f"{path} is missing {missing + ([] if 'seed' in config else ['seed'])}")
    return config
