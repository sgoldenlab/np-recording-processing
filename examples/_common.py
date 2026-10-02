"""Shared helpers of the example scripts (argument parsing, demo scratch folder)."""
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from np_ephys.config import load_config

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "config" / "example_config.toml"


def parse_args(description: str, extra=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", default=str(DEFAULT_CONFIG), help="TOML config (placeholder paths by default)")
    p.add_argument("--session", default=None, help="session name from the config (default: all)")
    p.add_argument("--demo", action="store_true",
                   help="run on small synthetic data in a temporary folder (no config paths used)")
    p.add_argument("--overwrite", action="store_true")
    if extra:
        extra(p)
    return p.parse_args()


def config(args) -> dict:
    return load_config(args.config)


def scratch(name: str) -> Path:
    return Path(tempfile.mkdtemp(prefix=f"np_ephys_{name}_"))
