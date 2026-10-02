"""Configuration loading: one TOML file drives the example scripts.

No import-time side effects; every function receives the config path or dict explicitly.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

try:  # python >= 3.11
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore

DEFAULT_NAMES = {
    # session folders (created lazily by the step that writes into them)
    "raw": "raw",
    "processed": "processed",
    "alignment": "alignment",
    "derived": "derived",
    "sync": "sync",
    "units": "units",
    "metrics": "metrics",
    # per-animal histology folders
    "downsampled": "downsampled",
    "registered": "registered",
    "stacks": "stacks",
    "tracks": "tracks",
}
DEFAULT_TEMPLATES = {
    "session_template": "{animal}/{session}",
    "histology_template": "{animal}/histology",
}


def load_config(path: str | Path) -> dict[str, Any]:
    """Read a TOML config and fill in default folder names/templates.

    ``[paths].root`` can be overridden with the environment variable ``NP_EPHYS_ROOT``
    (handy to keep machine-specific paths out of the committed file).
    """
    with open(path, "rb") as f:
        cfg = tomllib.load(f)
    paths = {**DEFAULT_TEMPLATES, **cfg.get("paths", {})}
    paths["names"] = {**DEFAULT_NAMES, **paths.get("names", {})}
    if os.environ.get("NP_EPHYS_ROOT"):
        paths["root"] = os.environ["NP_EPHYS_ROOT"]
    cfg["paths"] = paths
    cfg.setdefault("experiment", {}).setdefault("recordings", {})
    return cfg


def recordings_from_config(cfg: dict[str, Any], only: str | None = None) -> dict[str, dict]:
    """Return ``{session_name: {concatenate: bool, multiple_shanks: bool | list[bool]}}``.

    ``only`` restricts to a single session. Missing keys get defaults (False).
    """
    recs = {
        name: {"concatenate": False, "multiple_shanks": False, **props}
        for name, props in cfg["experiment"]["recordings"].items()
    }
    if only is not None:
        recs = {only: recs[only]}
    return recs
