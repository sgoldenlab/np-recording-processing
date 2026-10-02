"""Minimal, dependency-free reader for SpikeGLX ``.meta`` files.

Written from the public SpikeGLX metadata description (B. Karsh, HHMI Janelia,
https://billkarsh.github.io/SpikeGLX/). Independent re-implementation: no code was copied from
other packages.
"""
from __future__ import annotations

from pathlib import Path


def read_meta(meta_path: str | Path) -> dict[str, str]:
    """Parse ``key=value`` lines into a dict of strings (leading ``~`` of keys removed)."""
    meta_path = Path(meta_path)
    if not meta_path.exists():
        raise FileNotFoundError(f"no meta file: {meta_path}")
    meta: dict[str, str] = {}
    for line in meta_path.read_text().splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        meta[key.lstrip("~").strip()] = value.strip()
    return meta


def channel_counts(meta: dict[str, str]) -> tuple[int, int, int]:
    """(n_AP, n_LF, n_SY) channel counts from ``snsApLfSy``."""
    ap, lf, sy = (int(x) for x in meta["snsApLfSy"].split(","))
    return ap, lf, sy


def sample_rate(meta: dict[str, str]) -> float:
    """Imec sampling rate in Hz (``imSampRate``)."""
    return float(meta["imSampRate"])


def n_saved_channels(meta: dict[str, str]) -> int:
    """Total channels stored per sample in the binary file (AP+LF+SY)."""
    return int(meta["nSavedChans"])


def parse_geom_map(meta: dict[str, str]):
    """Parse ``snsGeomMap`` (NP2.0 / multi-shank probes).

    Header ``(probe_part,n_shank,shank_pitch_um,shank_width_um)`` followed by one
    ``(shank:x:y:used)`` entry per saved channel.

    Returns ``(n_shank, shank_pitch, shank_width, shank_ind, x, y, used)`` (arrays are numpy).
    """
    import numpy as np

    if "snsGeomMap" not in meta:
        raise KeyError("meta has no 'snsGeomMap' (older SpikeGLX / NP1.0); use the imro table instead")
    parts = meta["snsGeomMap"].split(")")
    header = parts[0].lstrip("(").split(",")
    n_shank, pitch, width = int(header[1]), float(header[2]), float(header[3])
    entries = [p.lstrip("(") for p in parts[1:] if p.strip("() ")]
    shank, x, y, used = [], [], [], []
    for e in entries:
        s, ex, ey, u = e.split(":")
        shank.append(int(s))
        x.append(float(ex))
        y.append(float(ey))
        used.append(int(u))
    return n_shank, pitch, width, np.array(shank), np.array(x), np.array(y), np.array(used)
