"""Finding and loading compressed raw recordings (``.cbin`` + ``.ch`` + ``.meta``).

Raw files may sit flat in ``raw/`` or in one sub-folder per SpikeGLX segment
(``{rec}_g0/``, ``{rec}_trials_g0/`` ...). Concatenated sessions need the per-segment layout.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from .layout import probe_from_name


def _natural_key(path: Path):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", str(path.name))]


def find_raw_files(
    raw_dir: str | Path,
    recording_name: str,
    concatenate: bool = False,
    order: list[str] | None = None,
    pattern: str = "*.ap.cbin",
) -> dict[int, list[Path]]:
    """Locate compressed AP files for a session, grouped by probe (imec index).

    Parameters
    ----------
    raw_dir : the session raw folder (``SessionPaths.raw``).
    recording_name : session name; files must contain it in the path.
    concatenate : if False, exactly one file per probe is expected (error otherwise);
        if True, several segment files per probe are returned in segment order.
    order : optional list of substrings giving the explicit segment order, e.g.
        ``['NP09_R1_g0', 'NP09_R1_trials_g0']``. Default: natural sort of the file names
        (``_g0_t0`` < ``_g0_t1`` < ``_g1_t0``).

    Returns
    -------
    ``{probe_index: [cbin paths]}`` sorted by probe, with segments in order
    (keyed by imec index).
    """
    raw_dir = Path(raw_dir)
    files = [f for f in raw_dir.rglob(pattern) if recording_name in str(f.relative_to(raw_dir))]
    groups: dict[int, list[Path]] = {}
    for f in files:
        probe = probe_from_name(f.name)
        if probe is None:
            raise ValueError(f"cannot find 'imec<N>' in file name {f.name}")
        groups.setdefault(probe, []).append(f)
    for probe, fl in groups.items():
        if order:
            def key(p: Path):
                s = str(p)
                for i, token in enumerate(order):
                    if token in s:
                        return (i, _natural_key(p))
                return (len(order), _natural_key(p))
            fl.sort(key=key)
        else:
            fl.sort(key=_natural_key)
        if not concatenate and len(fl) > 1:
            raise ValueError(
                f"{len(fl)} files for probe {probe} of '{recording_name}' but concatenate=False: "
                f"{[p.name for p in fl]}. Set concatenate=true in the config."
            )
    return dict(sorted(groups.items()))


def load_recording(files, include_sync: bool = False, concatenate: bool | None = None):
    """Load one ``.cbin`` (or ``.ap.bin``) or concatenate a list of segment files (SpikeInterface).

    ``include_sync=True`` keeps the digital sync channel as the LAST channel (needed for sync
    extraction; leave False for sorting).
    """
    import spikeinterface.full as si

    if isinstance(files, (str, Path)):
        files = [Path(files)]
    files = [Path(f) for f in files]
    if concatenate is None:
        concatenate = len(files) > 1
    if not concatenate and len(files) > 1:
        raise ValueError("several files given but concatenate=False")
    recs = []
    for f in files:
        if f.suffix == ".cbin":
            rec = si.read_cbin_ibl(cbin_file_path=f, load_sync_channel=include_sync, stream_name="ap")
        else:  # uncompressed SpikeGLX .bin fallback
            probe = probe_from_name(f.name)
            rec = si.read_spikeglx(f.parent, load_sync_channel=include_sync, stream_id=f"imec{probe}.ap")
        recs.append(rec)
    return recs[0] if len(recs) == 1 else si.concatenate_recordings(recs)


def get_segment_lengths(files, sampling_frequency: float | None = None) -> tuple[list[int], float]:
    """Number of samples of every segment file (needs SpikeInterface)."""
    lengths, fs = [], sampling_frequency
    for f in files:
        rec = load_recording(f, include_sync=False)
        lengths.append(int(rec.get_num_samples()))
        fs = float(rec.get_sampling_frequency())
    return lengths, fs


def get_segment_boundaries(files, cache_file: str | Path | None = None, overwrite: bool = False) -> list[dict]:
    """Segment boundaries in concatenated (global) sample coordinates, cached as JSON.

    Each entry: ``segment_idx, start_sample, end_sample (exclusive), length, file, sampling_frequency``.
    The pure computation lives in :func:`np_ephys.sync.segment_boundaries_from_lengths`.
    """
    from .sync import segment_boundaries_from_lengths

    if cache_file is not None and Path(cache_file).exists() and not overwrite:
        return json.loads(Path(cache_file).read_text())
    lengths, fs = get_segment_lengths(files)
    bounds = segment_boundaries_from_lengths(lengths, fs, [Path(f).name for f in files])
    if cache_file is not None:
        Path(cache_file).parent.mkdir(parents=True, exist_ok=True)
        Path(cache_file).write_text(json.dumps(bounds, indent=2))
    return bounds
