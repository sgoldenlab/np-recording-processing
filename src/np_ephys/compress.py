"""Lossless compression of SpikeGLX AP binaries with mtscomp (IBL).

``.ap.bin`` -> ``.ap.cbin`` + ``.ap.ch`` (+ a copy of ``.ap.meta``). Sampling rate and channel count are read
straight from the ``.meta`` file (so SpikeInterface is NOT needed); the sync channel is
included (``nSavedChans``), which is required for later sync extraction.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

from .layout import probe_from_name
from .spikeglx_meta import n_saved_channels, read_meta, sample_rate


def default_job_kwargs(fraction_of_cores: float = 0.8, chunk_duration: float = 5) -> dict:
    """``n_threads = round(0.8 * cpu_count)``, ``chunk_duration = 5 s``."""
    return dict(n_threads=max(1, round((os.cpu_count() or 1) * fraction_of_cores)),
                chunk_duration=chunk_duration)


def compress_bin(bin_file: str | Path, out_dir: str | Path, job_kwargs: dict | None = None,
                 overwrite: bool = False) -> Path:
    """Compress one ``*.ap.bin`` and copy its ``.meta`` into ``out_dir``. Returns the ``.cbin`` path."""
    from mtscomp import compress

    bin_file, out_dir = Path(bin_file), Path(out_dir)
    meta_file = bin_file.with_suffix(".meta")
    if not meta_file.exists():
        raise FileNotFoundError(f"meta file missing next to {bin_file}")
    meta = read_meta(meta_file)
    out_dir.mkdir(parents=True, exist_ok=True)
    cbin, ch = out_dir / bin_file.with_suffix(".cbin").name, out_dir / bin_file.with_suffix(".ch").name
    if cbin.exists() and not overwrite:
        return cbin
    compress(
        bin_file, out=cbin, outmeta=ch, sample_rate=sample_rate(meta),
        n_channels=n_saved_channels(meta), dtype="int16", **(job_kwargs or default_job_kwargs()),
    )
    shutil.copyfile(meta_file, out_dir / meta_file.name)
    return cbin


def find_ap_bins(segment_folder: str | Path) -> dict[int, Path]:
    """``{probe: ap.bin}`` inside a SpikeGLX run folder ``{rec}_g0`` (``{rec}_g0_imec{N}/``)."""
    out = {}
    for f in Path(segment_folder).rglob("*.ap.bin"):
        probe = probe_from_name(f.name)
        if probe is not None:
            out[probe] = f
    return dict(sorted(out.items()))


def compress_session(segment_folders, raw_target: str | Path, concatenate: bool = False,
                     job_kwargs: dict | None = None, overwrite: bool = False) -> list[Path]:
    """Compress every probe of every SpikeGLX segment folder into ``raw_target``.

    Non-concatenated sessions are written flat. Concatenated sessions get one sub-folder
    per segment (named like the SpikeGLX run folder) so that
    :func:`np_ephys.io_raw.find_raw_files` can order the segments.
    """
    raw_target = Path(raw_target)
    outputs = []
    for seg in segment_folders:
        seg = Path(seg)
        bins = find_ap_bins(seg)
        if not bins:
            raise FileNotFoundError(f"no *.ap.bin found below {seg}")
        dest = raw_target / seg.name if concatenate else raw_target
        for _probe, bin_file in bins.items():
            outputs.append(compress_bin(bin_file, dest, job_kwargs, overwrite))
    return outputs
