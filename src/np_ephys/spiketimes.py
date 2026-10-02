"""Per-unit spike time tables (polars) from a spike vector, restricted to sync windows.

Spike times are stored in samples and in seconds **since the start of the recording file**
(``sample / fs``), NOT re-zeroed to the first ping. Use :func:`np_ephys.sync.events_to_recording_time`
to bring event times to the same clock.
"""
from __future__ import annotations

import numpy as np
import polars as pl


def make_unit_key(recording_name: str, unit_id, probe_id: int | None = None, shank_id: int | None = None) -> str:
    """Unit identifier used to join tables across recordings.

    * ``probe_id`` None  -> ``{rec}_{unit}``                 (per-session files, quick path)
    * probe (+ shank)    -> ``{rec}_p{probe}_s{shank}_u{unit}`` (alignment tables; ``_s`` omitted if no shank)
    """
    if probe_id is None:
        return f"{recording_name}_{unit_id}"
    shank = "" if shank_id is None else f"_s{shank_id}"
    return f"{recording_name}_p{probe_id}{shank}_u{unit_id}"


def spikes_to_frame(sample_index, unit_index, unit_ids) -> pl.DataFrame:
    """Spike table with ``sample_index`` and the real ``unit_id``.

    ``unit_index`` is the position in ``unit_ids`` (as in ``sorting.to_spike_vector()``). Ids are
    not assumed to be 0..N-1 (units may have been removed or merged).
    """
    unit_ids = np.asarray(unit_ids)
    return pl.DataFrame(
        {"sample_index": np.asarray(sample_index, dtype=np.int64),
         "unit_id": unit_ids[np.asarray(unit_index, dtype=np.int64)].astype(np.int64)}
    )


def spikes_from_sorting(sorting) -> pl.DataFrame:
    """Spike table from a SpikeInterface sorting/analyzer.sorting (single segment). Lazy SI use."""
    sv = sorting.to_spike_vector()
    return spikes_to_frame(sv["sample_index"], sv["unit_index"], sorting.unit_ids)


def _bounds(window):
    if isinstance(window, dict):
        return window["start_sync"], window["end_sync"]
    return window


def filter_spikes_by_windows(spikes: pl.DataFrame, windows) -> pl.DataFrame:
    """Keep spikes with ``start <= sample_index <= end`` in ANY window.

    ``windows``: one ``(start, end)`` tuple, or a list of tuples / dicts with ``start_sync`` and
    ``end_sync`` (from :func:`np_ephys.sync.segment_sync_windows`). Empty list -> no filtering.
    """
    if isinstance(windows, tuple):
        windows = [windows]
    if not windows:
        return spikes
    mask = None
    for w in windows:
        s, e = _bounds(w)
        m = (pl.col("sample_index") >= s) & (pl.col("sample_index") <= e)
        mask = m if mask is None else (mask | m)
    return spikes.filter(mask)


def aggregate_spikes_per_unit(spikes: pl.DataFrame, sampling_frequency: float,
                              total_duration_samples: int | None = None) -> pl.DataFrame:
    """One row per unit: ``spike_times`` (samples), ``spike_times_sec``, ``num_spikes``, ``firing_rate`` (Hz).

    ``total_duration_samples`` is the analysed duration (window length, or summed window lengths
    for concatenated recordings); default: span between first and last spike.
    """
    if total_duration_samples is None:
        total_duration_samples = int(spikes["sample_index"].max() - spikes["sample_index"].min())
    fs = float(sampling_frequency)
    return (
        spikes.sort("sample_index")
        .group_by("unit_id", maintain_order=True)
        .agg(
            pl.col("sample_index").alias("spike_times"),
            (pl.col("sample_index") / fs).alias("spike_times_sec"),
            pl.len().alias("num_spikes"),
        )
        .with_columns((pl.col("num_spikes") / total_duration_samples * fs).alias("firing_rate"))
        .sort("unit_id")
    )


def windows_duration_samples(windows) -> int:
    """Total length (samples) of one or several windows."""
    if isinstance(windows, tuple):
        windows = [windows]
    return int(sum(_bounds(w)[1] - _bounds(w)[0] for w in windows))


def build_unit_table(spikes: pl.DataFrame, unit_metadata: pl.DataFrame, sampling_frequency: float,
                     windows=None, total_duration_samples: int | None = None) -> pl.DataFrame:
    """Filter spikes to ``windows``, aggregate per unit and inner-join unit metadata on ``unit_id``.

    Without windows the whole span is used (``total_duration_samples`` required for a true rate).
    """
    if windows is not None and total_duration_samples is None:
        total_duration_samples = windows_duration_samples(windows)
    filtered = filter_spikes_by_windows(spikes, windows) if windows is not None else spikes
    agg = aggregate_spikes_per_unit(filtered, sampling_frequency, total_duration_samples)
    md = unit_metadata.with_columns(pl.col("unit_id").cast(pl.Int64))
    return md.join(agg, on="unit_id", how="inner")
