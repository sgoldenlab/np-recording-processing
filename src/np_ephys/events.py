"""Parse Raspberry Pi event logs / video frame logs and align them to video and recording time.

Input files (see ``hardware/README.md`` for how they are produced):

* ``{rec}_events.csv``  columns ``time_ms, time_s, event, gas_concentration[, video_frame]``
  (3-column ``time_s, event, gas_concentration`` also accepted). One ``Gas measurement`` row is
  written per Arduino ping; button presses are ``Event_NN`` / ``EVENT_<n>_<label>``;
  ``TONESET_<n>_<label>`` rows mark tones. Times are on the ARDUINO clock (zero = start button).
* ``{rec}_frames.csv``  columns ``Frame_ID, Hardware_Timestamp_ns, System_Arrival_Time_s``.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from .sync import events_to_recording_time, match_ping_counts

_ALIASES = {
    "time_ms": ("time_ms",),
    "time_s": ("time_s",),
    "event": ("event",),
    "gas_concentration": ("gas_concentration", "sensor_value", "gas_measure"),
    "video_frame": ("video_frame",),
}


def _num(col: str) -> pl.Expr:
    return pl.col(col).cast(pl.Utf8).str.strip_chars().str.replace(r"s$", "").cast(pl.Float64, strict=False)


def load_events(events_file) -> pl.DataFrame:
    """Read an events CSV into ``time_ms, time_s, event, gas_concentration[, video_frame]``.

    Unparseable numbers (``N/A``, empty) become null. ``time_s`` is derived from ``time_ms`` when
    present, else read from ``time_s``; ``time_ms`` is derived from ``time_s`` otherwise.
    """
    raw = pl.read_csv(events_file, infer_schema_length=0, truncate_ragged_lines=True)
    lower = {c: c.strip().lower() for c in raw.columns}
    raw = raw.rename(lower)
    mapping = {}
    for target, names in _ALIASES.items():
        for n in names:
            if n in raw.columns:
                mapping[n] = target
                break
    if "event" not in mapping.values():
        # headerless / unusual header: fall back on position
        cols = raw.columns
        pos = ["time_s", "event", "gas_concentration"] if len(cols) == 3 else \
            ["time_ms", "time_s", "event", "gas_concentration", "video_frame"][: len(cols)]
        mapping = dict(zip(cols, pos))
    df = raw.rename(mapping)
    if "time_ms" in df.columns:
        df = df.with_columns(_num("time_ms").alias("time_ms")).with_columns((pl.col("time_ms") / 1000.0).alias("time_s"))
    else:
        df = df.with_columns(_num("time_s").alias("time_s")).with_columns((pl.col("time_s") * 1000.0).alias("time_ms"))
    if "gas_concentration" not in df.columns:
        df = df.with_columns(pl.lit(None, dtype=pl.Float64).alias("gas_concentration"))
    df = df.with_columns(_num("gas_concentration").alias("gas_concentration"))
    keep = ["time_ms", "time_s", "event", "gas_concentration"] + (["video_frame"] if "video_frame" in df.columns else [])
    return df.select(keep).drop_nulls(subset=["time_s"])


def parse_event_labels(events: pl.DataFrame) -> pl.DataFrame:
    """Add ``event_type`` (gas | event | toneset | other), ``event_num`` and ``event_label``.

    * ``Gas measurement``       -> gas, label ``gas_measurement``
    * ``EVENT_1_ISO_on``        -> event, num 1, label ``ISO on`` (underscores -> spaces)
    * ``Event_03``  -> event, num 3, label ``unlabeled``
    * ``TONESET_0_tone_0``      -> toneset, num 0, label ``tone_0``
    """
    ev = pl.col("event")
    is_gas = ev == "Gas measurement"
    is_event = ev.str.to_uppercase().str.starts_with("EVENT_")
    is_tone = ev.str.to_uppercase().str.starts_with("TONESET_")
    label = (
        pl.when(is_gas).then(pl.lit("gas_measurement"))
        .when(is_event).then(ev.str.extract(r"(?i)^EVENT_\d+_?(.*)$", 1).str.replace_all("_", " ").str.strip_chars())
        .when(is_tone).then(ev.str.extract(r"(?i)^TONESET_\d+_(.*)$", 1))
        .otherwise(ev)
    )
    out = events.with_columns(
        pl.when(is_gas).then(pl.lit("gas")).when(is_event).then(pl.lit("event"))
        .when(is_tone).then(pl.lit("toneset")).otherwise(pl.lit("other")).alias("event_type"),
        ev.str.extract(r"(?i)^(?:EVENT|TONESET)_(\d+)", 1).cast(pl.Int32).alias("event_num"),
        label.alias("event_label"),
    )
    return out.with_columns(
        pl.when(pl.col("event_label").is_null() | (pl.col("event_label") == "")).then(pl.lit("unlabeled"))
        .otherwise(pl.col("event_label")).alias("event_label")
    )


def load_frames(frames_file) -> pl.DataFrame:
    """Read the frame log; adds ``frame_time_s`` = hardware time since the first frame."""
    df = pl.read_csv(frames_file)
    c = df.columns
    df = df.rename({c[0]: "frame_id", c[1]: "hw_timestamp_ns", c[2]: "system_time_s"})
    first = df["hw_timestamp_ns"][0]
    return df.with_columns(((pl.col("hw_timestamp_ns") - first) / 1e9).alias("frame_time_s"))


def create_sync_frame_mapping(events: pl.DataFrame, frames: pl.DataFrame) -> pl.DataFrame:
    """For each ping (``Gas measurement`` row) find the closest video frame.

    Assumes frame time 0 (first frame) and Arduino time 0 (start button) coincide; the residual
    is reported as ``time_offset_ms`` so you can check it.
    """
    sync = events.filter(pl.col("event_type") == "gas")["time_s"].to_numpy()
    ft = frames["frame_time_s"].to_numpy()
    fid = frames["frame_id"].to_numpy()
    idx = _nearest_index(ft, sync)
    return pl.DataFrame({
        "sync_time_s": sync,
        "frame_id": fid[idx].astype(int),
        "frame_time_s": ft[idx],
        "time_offset_ms": (ft[idx] - sync) * 1000.0,
    })


def _nearest_index(sorted_values: np.ndarray, query: np.ndarray) -> np.ndarray:
    pos = np.clip(np.searchsorted(sorted_values, query), 1, len(sorted_values) - 1)
    left, right = sorted_values[pos - 1], sorted_values[pos]
    return np.where(np.abs(query - left) <= np.abs(right - query), pos - 1, pos)


def align_events(events: pl.DataFrame, sync_frame_map: pl.DataFrame | None = None) -> pl.DataFrame:
    """Labeled (non-gas) events with ``sync_time_s`` and the closest ping's video frame.

    Output columns: ``event_label, event_type, event_num, event, sync_time_s, gas_concentration,
    video_frame, video_time_s`` (``sync_time_s`` is on the Arduino clock).
    """
    labeled = events.filter(pl.col("event_type") != "gas")
    out = labeled.select("event_label", "event_type", "event_num", "event",
                         pl.col("time_s").alias("sync_time_s"), "gas_concentration")
    if sync_frame_map is None or sync_frame_map.height == 0:
        return out.with_columns(pl.lit(None, dtype=pl.Int64).alias("video_frame"),
                                pl.lit(None, dtype=pl.Float64).alias("video_time_s"))
    st = sync_frame_map["sync_time_s"].to_numpy()
    idx = _nearest_index(st, out["sync_time_s"].to_numpy())
    return out.with_columns(
        pl.Series("video_frame", sync_frame_map["frame_id"].to_numpy()[idx].astype("int64")),
        pl.Series("video_time_s", sync_frame_map["frame_time_s"].to_numpy()[idx]),
    )


def add_recording_time(events: pl.DataFrame, rec_ping_times, rec_min_time_s: float | None = None,
                       on_mismatch: str = "truncate") -> pl.DataFrame:
    """Add ``recording_time_s`` (time in the neural recording) to a parsed events table.

    Pairs the logged ``Gas measurement`` times with ``rec_ping_times`` (from
    :func:`np_ephys.sync.get_recording_sync`) and interpolates every row, pings included.
    """
    log = events.filter(pl.col("event_type") == "gas")["time_s"].to_numpy()
    log, rec = match_ping_counts(log, rec_ping_times, rec_min_time_s, on_mismatch)
    new = events_to_recording_time(events["time_s"].to_numpy(), log, rec)
    return events.with_columns(pl.Series("recording_time_s", new))
