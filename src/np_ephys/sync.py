"""Sync-ping extraction and time-base conversion.

Hardware (see ``hardware/README.md``): an Arduino drives a ~1 Hz 50 % duty square wave into the
digital sync input of the acquisition system. SpikeGLX stores it as the LAST channel of the AP
stream. Every rising edge ("ping") is also logged by the Raspberry Pi (``Gas measurement`` rows of
``events.csv``) with the Arduino clock, so ping *k* in the recording = log row *k*.

Pure-numpy functions are tested on synthetic arrays; functions touching raw files import
SpikeInterface lazily.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------- edge detection


def detect_rising_edges(trace, threshold: float | None = None, height: float = 0.5,
                        prominence: float = 0.5) -> np.ndarray:
    """Indices ``i`` of rising edges between sample ``i`` and ``i+1`` of a 1-D sync trace.

    * ``threshold`` given: ``trace[i] < threshold <= trace[i+1]`` (the lab used 40 for raw int16 data).
    * ``threshold=None`` : peaks of ``np.diff(trace)`` via ``scipy.signal.find_peaks(height, prominence)``
      (defaults ``0.5`` / ``0.5``).
    """
    x = np.asarray(trace).astype("float64").ravel()
    if x.size < 2:
        return np.array([], dtype=int)
    if threshold is not None:
        return np.where((x[:-1] < threshold) & (x[1:] >= threshold))[0]
    from scipy.signal import find_peaks

    idx, _ = find_peaks(np.diff(x), height=height, prominence=prominence)
    return idx


def get_sync_edges(recording, threshold: float | None = None, time_range: tuple[float, float] | None = None,
                   chunk_duration_s: float = 10.0, channel_index: int = -1,
                   find_peaks_kwargs: dict | None = None):
    """Rising edges of one channel of a recording-like object, processed in chunks.

    ``recording`` needs ``get_num_samples()``, ``get_sampling_frequency()`` and
    ``get_traces(start_frame=, end_frame=, return_scaled=False)`` (any SpikeInterface recording).
    The sync channel is ``channel_index`` (default: last).

    Chunks are read with a 2-sample margin on both sides and every edge is assigned to the chunk
    that contains its first sample, so edges falling exactly on a chunk boundary are neither lost
    nor duplicated.

    Returns ``(samples, times_s)``; ``times_s = samples / fs + t_start``.
    """
    fs = float(recording.get_sampling_frequency())
    n = int(recording.get_num_samples())
    lo, hi = 0, n
    if time_range is not None:
        lo = max(0, int(time_range[0] * fs))
        hi = min(n, int(time_range[1] * fs)) if time_range[1] is not None else n
    chunk = max(int(chunk_duration_s * fs), 8)
    kw = find_peaks_kwargs or {}
    found = []
    for s in range(lo, hi, chunk):
        e = min(s + chunk, hi)
        a, b = max(0, s - 2), min(n, e + 2)
        traces = np.asarray(recording.get_traces(start_frame=a, end_frame=b, return_scaled=False))
        x = traces[:, channel_index] if traces.ndim == 2 else traces
        idx = detect_rising_edges(x, threshold, **kw) + a
        found.append(idx[(idx >= s) & (idx < e)])
    samples = np.concatenate(found).astype(np.int64) if found else np.array([], dtype=np.int64)
    t0 = 0.0
    if hasattr(recording, "get_start_time"):
        try:
            t0 = float(recording.get_start_time() or 0.0)
        except Exception:  # noqa: BLE001 - signature varies by SpikeInterface version
            t0 = 0.0
    return samples, samples / fs + t0


# --------------------------------------------------------------------------- windows / periods


def extract_sync_periods(ping_times, min_gap: float = 60.0, min_duration: float = 30.0):
    """Split ping times into ``(start, end)`` periods wherever the gap exceeds ``min_gap`` s.

    Periods shorter than ``min_duration`` s are dropped.
    """
    t = np.asarray(ping_times, dtype=float)
    if t.size == 0:
        return []
    cut = np.where(np.diff(t) > min_gap)[0]
    starts = np.r_[0, cut + 1]
    ends = np.r_[cut, t.size - 1]
    return [(float(t[s]), float(t[e])) for s, e in zip(starts, ends) if t[e] - t[s] >= min_duration]


def experiment_window(ping_samples, end_index: int = -2):
    """``(start_sample, end_sample)`` of the experiment from sync pings, or ``None`` if < 2 pings.

    Convention of the lab: start = first ping, end = SECOND-TO-LAST ping (the last ping can be the
    stop-button artefact). Change ``end_index`` to -1 to use the last ping.
    """
    p = np.asarray(ping_samples)
    if p.size < 2:
        return None
    return int(p[0]), int(p[end_index])


# --------------------------------------------------------------------------- concatenated segments


def segment_boundaries_from_lengths(lengths, sampling_frequency: float, files=None) -> list[dict]:
    """Cumulative segment boundaries in global (concatenated) sample coordinates."""
    out, cum = [], 0
    for i, ln in enumerate(lengths):
        out.append(dict(segment_idx=i, start_sample=cum, end_sample=cum + int(ln), length=int(ln),
                        file=None if files is None else str(files[i]),
                        sampling_frequency=float(sampling_frequency)))
        cum += int(ln)
    return out


def map_segment_sync_to_global(ping_samples_local, segment_boundaries, segment_idx: int, end_index: int = -2):
    """``(start_global, end_global)`` of a segment's sync window, or ``(None, None)`` if < 2 pings."""
    win = experiment_window(ping_samples_local, end_index)
    if win is None:
        return None, None
    off = segment_boundaries[segment_idx]["start_sample"]
    return off + win[0], off + win[1]


def segment_sync_windows(segment_boundaries, local_pings_per_segment, end_index: int = -2) -> list[dict]:
    """Global-coordinate experiment windows for all segments (segments without sync are skipped).

    Each dict has ``segment_idx, start_sync, end_sync, duration, file``; feed it to
    :func:`np_ephys.spiketimes.filter_spikes_by_windows` (``windows`` = list of dicts or tuples).
    """
    wins = []
    for i, pings in enumerate(local_pings_per_segment):
        s, e = map_segment_sync_to_global(pings, segment_boundaries, i, end_index)
        if s is None:
            warnings.warn(f"no valid sync pings in segment {i}; segment excluded from windows")
            continue
        wins.append(dict(segment_idx=i, start_sync=s, end_sync=e, duration=e - s,
                         file=segment_boundaries[i].get("file")))
    return wins


# --------------------------------------------------------------------------- cache files


def sync_cache_paths(out_dir, probe: int, segment: int | None = None, range_index: int | None = None,
                     threshold: float | None = None) -> tuple[Path, Path]:
    """Cache file names that encode probe, segment, range AND threshold.

    ``ping_samples_probe0_seg1_thr40_range0.npy``; ``thr`` is ``auto`` for the diff/find_peaks mode.
    Different thresholds or segments therefore never share a cache file.
    """
    thr = "auto" if threshold is None else (f"{threshold:g}")
    tag = f"probe{probe}"
    if segment is not None:
        tag += f"_seg{segment}"
    tag += f"_thr{thr}"
    if range_index is not None:
        tag += f"_range{range_index}"
    out_dir = Path(out_dir)
    return out_dir / f"ping_samples_{tag}.npy", out_dir / f"ping_times_{tag}.npy"


def get_recording_sync(raw_files, out_dir, probe: int, segment: int | None = None, concatenate: bool = False,
                       threshold: float | None = None, time_ranges=None, range_index: int | None = None,
                       overwrite: bool = False, chunk_duration_s: float = 10.0):
    """Extract (or load cached) sync pings from raw ``.cbin`` file(s). Needs SpikeInterface.

    ``time_ranges=[(start_s, end_s_or_None), ...]`` + ``range_index`` extract one period of a
    recording (e.g. ``[(0, 4320), (4320, None)]``); results are cached per range.
    Returns ``(ping_samples, ping_times)`` or ``(None, None)`` if no ping was found.
    """
    from .io_raw import load_recording

    if (time_ranges is None) != (range_index is None):
        raise ValueError("time_ranges and range_index must be given together")
    sp, tp = sync_cache_paths(out_dir, probe, segment, range_index, threshold)
    if sp.exists() and tp.exists() and not overwrite:
        return np.load(sp), np.load(tp)
    rec = load_recording(raw_files, include_sync=True, concatenate=concatenate)
    window = None
    if time_ranges is not None:
        start, end = time_ranges[range_index]
        window = (start, rec.get_total_duration() if end is None else end)
    samples, times = get_sync_edges(rec, threshold=threshold, time_range=window,
                                    chunk_duration_s=chunk_duration_s)
    if samples.size == 0:
        return None, None
    sp.parent.mkdir(parents=True, exist_ok=True)
    np.save(sp, samples)
    np.save(tp, times)
    return samples, times


# --------------------------------------------------------------------------- event clock -> recording clock


def match_ping_counts(log_times, rec_times, rec_min_time_s: float | None = None, on_mismatch: str = "truncate"):
    """Pair the Arduino/Pi ping log with the pings detected in the recording.

    ``rec_min_time_s`` drops detected pings earlier than that (artefact pings at acquisition
    start; 50 s is a typical value). If the counts still differ: ``'truncate'`` keeps the first
    ``min(n_log, n_rec)`` pairs (assumes pings were missed/added at the END), ``'raise'`` raises.
    """
    log = np.asarray(log_times, dtype=float)
    rec = np.asarray(rec_times, dtype=float)
    if rec_min_time_s is not None:
        rec = rec[rec >= rec_min_time_s]
    if log.size != rec.size:
        if on_mismatch == "raise":
            raise ValueError(f"{log.size} logged pings vs {rec.size} detected pings")
        n = min(log.size, rec.size)
        warnings.warn(f"ping count mismatch (log {log.size}, recording {rec.size}); using first {n}")
        log, rec = log[:n], rec[:n]
    return log, rec


def ping_clock_fit(log_times, rec_times) -> dict:
    """Linear fit ``rec = slope * log + offset`` and its worst residual (QC for the pairing)."""
    log = np.asarray(log_times, dtype=float)
    rec = np.asarray(rec_times, dtype=float)
    slope, offset = np.polyfit(log, rec, 1)
    resid = rec - (slope * log + offset)
    return dict(slope=float(slope), offset=float(offset), max_abs_residual_s=float(np.max(np.abs(resid))),
                n=int(log.size))


def events_to_recording_time(event_times_s, log_ping_times_s, rec_ping_times_s) -> np.ndarray:
    """Convert event times on the Arduino/Pi clock to recording time (seconds).

    Piecewise-linear interpolation between the paired pings absorbs clock drift between the
    Arduino and the acquisition clock. Events before the first / after the last ping are
    extrapolated with the global ping-to-ping slope, anchored at the nearest ping.

    ``log_ping_times_s`` and ``rec_ping_times_s`` must be paired one-to-one and increasing
    (use :func:`match_ping_counts`).
    """
    ev = np.asarray(event_times_s, dtype=float)
    log = np.asarray(log_ping_times_s, dtype=float)
    rec = np.asarray(rec_ping_times_s, dtype=float)
    if log.size != rec.size or log.size < 2:
        raise ValueError("need >= 2 paired pings of equal length")
    if np.any(np.diff(log) <= 0):
        raise ValueError("logged ping times must be strictly increasing")
    out = np.interp(ev, log, rec)
    slope = ping_clock_fit(log, rec)["slope"]
    before, after = ev < log[0], ev > log[-1]
    out[before] = rec[0] + slope * (ev[before] - log[0])
    out[after] = rec[-1] + slope * (ev[after] - log[-1])
    return out
