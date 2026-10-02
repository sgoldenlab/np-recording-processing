"""Preprocessing of AP recordings for postprocessing, and LFP extraction (SpikeInterface, lazy)."""
from __future__ import annotations

from pathlib import Path

DEFAULT_SAVING_JOB_KWARGS = dict(n_jobs=6, chunk_duration="1s", mp_context="spawn", max_threads_per_worker=4)


def kilosort_like_preprocess(rec, freq_min: float = 300.0, freq_max: float = 6000.0):
    """Preprocessing chain used for waveform/metric extraction (Kilosort sorts the RAW data).

    per shank (``group``): bandpass(300-6000 Hz) -> phase_shift -> aggregate -> detect and interpolate
    bad channels -> per shank high-pass spatial filter (``n_channel_pad=10`` fallback for narrow shanks)
    -> aggregate.
    """
    import spikeinterface.full as si

    split = rec.split_by("group")
    filt = si.bandpass_filter(split, freq_min=freq_min, freq_max=freq_max)
    filt = si.phase_shift(recording=filt)
    pre = si.aggregate_channels(filt)
    bad_ids, _ = si.detect_bad_channels(recording=pre)
    pre = si.interpolate_bad_channels(recording=pre, bad_channel_ids=bad_ids)
    split = pre.split_by("group")
    try:
        filt = si.highpass_spatial_filter(recording=split)
    except AssertionError:  # shank narrower than the default padding
        filt = si.highpass_spatial_filter(recording=split, n_channel_pad=10)
    return si.aggregate_channels(filt)


def save_recording_zarr(rec, zarr_path: Path, saving_job_kwargs: dict | None = None):
    """Save to compressed zarr (LZMA) and return the saved recording."""
    from numcodecs import LZMA

    return rec.save_to_zarr(folder=zarr_path, backend_options=dict(compressor=LZMA()), overwrite=True,
                            verbose=True, **(saving_job_kwargs or DEFAULT_SAVING_JOB_KWARGS))


def make_lfp(rec, freq_min: float = 1.0, freq_max: float = 300.0, decimate_factor: int = 10):
    """LFP from the 30 kHz AP stream: bandpass 1-300 Hz, decimate by 10 (3 kHz)."""
    import spikeinterface.full as si

    return si.decimate(si.bandpass_filter(rec, freq_min=freq_min, freq_max=freq_max), decimate_factor)


def make_lfp_resampled(rec, freq_min: float = 0.5, freq_max: float = 250.0, resample_rate: float = 250.0):
    """Experimental LFP variant: 3rd-order bandpass 0.5-250 Hz (reflect padding) -> phase_shift -> resample 250 Hz."""
    import spikeinterface.full as si

    lfp = si.bandpass_filter(rec, freq_min=freq_min, freq_max=freq_max, margin_ms=1500.0, filter_order=3,
                             dtype="float32", add_reflect_padding=True)
    return si.resample(si.phase_shift(lfp), resample_rate=resample_rate, margin_ms=1000)
