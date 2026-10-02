"""Folder layout: pure path arithmetic, folders are created only when something is written.

Nothing here creates a folder when you resolve a path. A step creates the folder it writes to (``ensure_dir`` /
``ensure_parent``), so a session that never produces e.g. LFP or alignment data never has an empty ``alignment/`` folder.

Default layout (every name and the two templates are configurable, see ``config/example_config.toml``)::

    <root>/<animal>/
        histology/                        HistologyPaths: downsampled/ registered/ stacks/ tracks/
        <animal>_R<n>/                    SessionPaths (one recording session)
            raw/                          compressed recordings (.cbin/.ch/.meta)
            processed/                    kilosort4_probe<N>/, *.zarr analyzers, metrics/, lfp/
            alignment/                    ibl_alignment_SI_probe<N>/, channel tables, resolved regions
            derived/                      sync/ (ping caches), units/ (per-unit tables)

To use another hierarchy, change ``session_template`` / ``histology_template`` (e.g. ``"{session}"`` for a flat
layout) and the folder names; code that uses ``SessionPaths`` properties does not need to change.
Session names follow ``{animal}_R{n}``; the animal id is everything before the first underscore.
"""
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass, field
from pathlib import Path

from .config import DEFAULT_NAMES

SESSION_FOLDERS = ("raw", "processed", "alignment", "derived")
HISTOLOGY_FOLDERS = ("downsampled", "registered", "stacks", "tracks")


def animal_of(session: str) -> str:
    """``'NP09_R1' -> 'NP09'``."""
    return session.split("_")[0]


def probe_from_name(name: str) -> int | None:
    """Extract the SpikeGLX imec index from a file/folder name (``...imec1...`` -> 1)."""
    m = re.search(r"imec(\d+)", name)
    return int(m.group(1)) if m else None


def ensure_dir(path: str | Path) -> Path:
    """Create ``path`` (and parents) if needed and return it. Call this where a step writes a folder."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def ensure_parent(file_path: str | Path) -> Path:
    """Create the parent folder of an output file and return the file path."""
    p = Path(file_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _names(names: dict | None) -> dict:
    return {**DEFAULT_NAMES, **(names or {})}


@dataclass(frozen=True)
class SessionPaths:
    """Standard paths of one recording session. Resolving a path never creates anything."""

    session: str
    root: Path                       # the session folder itself
    names: dict = field(default_factory=dict)   # overrides of DEFAULT_NAMES

    @classmethod
    def from_config(cls, cfg: dict, session: str) -> SessionPaths:
        """``root = paths.root / session_template.format(animal=..., session=...)``."""
        p = cfg["paths"]
        rel = p["session_template"].format(animal=animal_of(session), session=session)
        return cls(session=session, root=Path(p["root"]) / rel, names=p.get("names", {}))

    @classmethod
    def from_root(cls, session: str, data_root: str | Path, template: str = "{animal}/{session}",
                  names: dict | None = None) -> SessionPaths:
        rel = template.format(animal=animal_of(session), session=session)
        return cls(session=session, root=Path(data_root) / rel, names=names or {})

    def _n(self, key: str) -> str:
        return _names(self.names)[key]

    @property
    def animal(self) -> str:
        return animal_of(self.session)

    @property
    def raw(self) -> Path:
        return self.root / self._n("raw")

    @property
    def processed(self) -> Path:
        return self.root / self._n("processed")

    @property
    def alignment(self) -> Path:
        return self.root / self._n("alignment")

    @property
    def derived(self) -> Path:
        return self.root / self._n("derived")

    @property
    def sync_data(self) -> Path:
        return self.derived / self._n("sync")

    @property
    def unit_tables(self) -> Path:
        """Per-unit tables restricted to the experiment window (output of ``examples/05_spiketimes.py``)."""
        return self.derived / self._n("units")

    @property
    def quality_metrics(self) -> Path:
        return self.processed / self._n("metrics")

    def kilosort_dir(self, probe: int) -> Path:
        return self.processed / f"kilosort4_probe{probe}"

    def preprocessed_zarr(self, probe: int, concatenated: bool = False) -> Path:
        suffix = "_concatenated" if concatenated else ""
        return self.processed / f"preprocessed_recording_probe{probe}{suffix}.zarr"

    def analyzer_zarr(self, probe: int, kind: str = "") -> Path:
        """kind: '' (initial), 'clean', 'quick', 'slay' -> ``sorting_analyzer[_kind]_probe{N}.zarr``."""
        mid = f"_{kind}" if kind else ""
        return self.processed / f"sorting_analyzer{mid}_probe{probe}.zarr"

    def lfp_zarr(self, probe: int) -> Path:
        return self.processed / "lfp" / f"{self.session}_probe{probe}.zarr"

    def ibl_export(self, probe: int) -> Path:
        return self.alignment / f"ibl_alignment_SI_probe{probe}"

    def existing(self) -> list[Path]:
        """Standard session folders that currently exist (handy to see what a session already has)."""
        return [d for d in (self.raw, self.processed, self.alignment, self.derived) if d.is_dir()]


@dataclass(frozen=True)
class HistologyPaths:
    """Per-animal histology folders (same lazy rules as :class:`SessionPaths`)."""

    animal: str
    root: Path                       # the animal's histology folder
    names: dict = field(default_factory=dict)

    @classmethod
    def from_config(cls, cfg: dict, animal: str) -> HistologyPaths:
        p = cfg["paths"]
        rel = p["histology_template"].format(animal=animal)
        return cls(animal=animal, root=Path(p["root"]) / rel, names=p.get("names", {}))

    @classmethod
    def from_root(cls, animal: str, data_root: str | Path, template: str = "{animal}/histology",
                  names: dict | None = None) -> HistologyPaths:
        return cls(animal=animal, root=Path(data_root) / template.format(animal=animal), names=names or {})

    def _n(self, key: str) -> str:
        return _names(self.names)[key]

    @property
    def downsampled(self) -> Path:
        """Arivis output: downsampled, rotated, not registered."""
        return self.root / self._n("downsampled")

    @property
    def registered(self) -> Path:
        """brainreg output folder(s)."""
        return self.root / self._n("registered")

    @property
    def stacks(self) -> Path:
        """Optional extra stacks (ImageJ flips/rotations/trims, NRRD reslices)."""
        return self.root / self._n("stacks")

    @property
    def tracks(self) -> Path:
        """Lasagna ``*_pts.csv`` files."""
        return self.root / self._n("tracks")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description="Show where session/histology folders live; with --create make only the folders you name.")
    ap.add_argument("--root", required=True, help="data root (parent of the animal folders)")
    ap.add_argument("--session", help="session name, e.g. NP09_R1")
    ap.add_argument("--animal", help="animal id for the histology folders (default: animal of --session)")
    ap.add_argument("--session-template", default="{animal}/{session}")
    ap.add_argument("--histology-template", default="{animal}/histology")
    ap.add_argument("--make", nargs="*", default=[], metavar="FOLDER",
                    help=f"which folders to create with --create: session {SESSION_FOLDERS}, histology {HISTOLOGY_FOLDERS}")
    ap.add_argument("--create", action="store_true", help="create the folders named in --make (default: only print)")
    a = ap.parse_args(argv)
    todo: list[Path] = []
    if a.session:
        sp = SessionPaths.from_root(a.session, a.root, a.session_template)
        todo += [getattr(sp, k) for k in SESSION_FOLDERS if k in a.make]
    animal = a.animal or (animal_of(a.session) if a.session else None)
    if animal:
        hp = HistologyPaths.from_root(animal, a.root, a.histology_template)
        todo += [getattr(hp, k) for k in HISTOLOGY_FOLDERS if k in a.make]
    unknown = set(a.make) - set(SESSION_FOLDERS) - set(HISTOLOGY_FOLDERS)
    if unknown:
        ap.error(f"unknown folder name(s): {sorted(unknown)}")
    if not todo:
        if a.session:
            sp = SessionPaths.from_root(a.session, a.root, a.session_template)
            print(f"session root: {sp.root}")
            for k in SESSION_FOLDERS:
                print(f"  {k:10s} {getattr(sp, k)}  {'(exists)' if getattr(sp, k).is_dir() else ''}")
        if animal:
            hp = HistologyPaths.from_root(animal, a.root, a.histology_template)
            print(f"histology root: {hp.root}")
            for k in HISTOLOGY_FOLDERS:
                print(f"  {k:10s} {getattr(hp, k)}  {'(exists)' if getattr(hp, k).is_dir() else ''}")
        return
    for d in todo:
        if a.create:
            ensure_dir(d)
        print(("created " if a.create else "would create ") + str(d))


if __name__ == "__main__":
    main()
