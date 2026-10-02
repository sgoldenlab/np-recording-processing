import numpy as np
import pytest

from np_ephys import sync


class FakeRec:
    """Duck-typed recording: 1 sync channel (last) + 1 dummy."""

    def __init__(self, sync_trace, fs=1000.0):
        self.sync = np.asarray(sync_trace, dtype="int16")
        self.fs = fs

    def get_num_samples(self):
        return self.sync.size

    def get_sampling_frequency(self):
        return self.fs

    def get_traces(self, start_frame=None, end_frame=None, return_scaled=False):
        s = self.sync[start_frame:end_frame]
        return np.stack([np.zeros_like(s), s], axis=1)


def square(n, period, high_len, first, level=64):
    x = np.zeros(n, dtype="int16")
    for s in range(first, n, period):
        x[s:s + high_len] = level
    return x


def test_detect_modes_agree():
    x = square(5000, 1000, 500, 100)
    a = sync.detect_rising_edges(x, threshold=40)
    b = sync.detect_rising_edges(x, threshold=None)
    assert list(a) == list(b) == [99, 1099, 2099, 3099, 4099]


@pytest.mark.parametrize("chunk_s", [0.1, 0.25, 1.0, 10.0])
@pytest.mark.parametrize("thr", [40, None])
def test_chunk_boundaries_do_not_lose_or_duplicate(chunk_s, thr):
    # periods chosen so edges fall exactly on chunk boundaries for some chunk sizes
    x = square(8000, 250, 125, 100)
    rec = FakeRec(x)
    ref = sync.detect_rising_edges(x, thr)
    samples, times = sync.get_sync_edges(rec, threshold=thr, chunk_duration_s=chunk_s)
    assert list(samples) == list(ref)
    assert np.allclose(times, samples / 1000.0)


def test_time_range():
    x = square(8000, 1000, 500, 100)
    s, _ = sync.get_sync_edges(FakeRec(x), threshold=40, time_range=(2.0, 5.0), chunk_duration_s=1.0)
    assert list(s) == [2099, 3099, 4099]
    s, _ = sync.get_sync_edges(FakeRec(x), threshold=40, time_range=(5.0, None))
    assert list(s) == [5099, 6099, 7099]


def test_periods_and_window():
    t = np.r_[np.arange(60, 160, 1.0), np.arange(300, 400, 1.0), [500.0]]
    assert sync.extract_sync_periods(t, min_gap=60, min_duration=30) == [(60.0, 159.0), (300.0, 399.0)]
    assert sync.extract_sync_periods([]) == []
    assert sync.experiment_window([10, 20, 30, 40]) == (10, 30)
    assert sync.experiment_window([10, 20, 30, 40], end_index=-1) == (10, 40)
    assert sync.experiment_window([10]) is None


def test_segments_global_mapping():
    b = sync.segment_boundaries_from_lengths([54000, 54000], 30000.0, ["a.cbin", "b.cbin"])
    assert b[1]["start_sample"] == 54000 and b[1]["end_sample"] == 108000
    assert sync.map_segment_sync_to_global([2000, 3000, 50000, 52000], b, 1) == (56000, 104000)
    assert sync.map_segment_sync_to_global([5], b, 0) == (None, None)
    with pytest.warns(UserWarning):
        w = sync.segment_sync_windows(b, [[1000, 2000, 5000, 6000], [7]])
    assert len(w) == 1 and w[0]["start_sync"] == 1000 and w[0]["end_sync"] == 5000


def test_cache_names_encode_everything(tmp_path):
    a = sync.sync_cache_paths(tmp_path, 0, segment=0, threshold=40)[0].name
    b = sync.sync_cache_paths(tmp_path, 0, segment=1, threshold=40)[0].name
    c = sync.sync_cache_paths(tmp_path, 0, segment=0, threshold=None)[0].name
    d = sync.sync_cache_paths(tmp_path, 0, segment=0, threshold=40, range_index=1)[1].name
    assert len({a, b, c}) == 3
    assert a == "ping_samples_probe0_seg0_thr40.npy"
    assert c.endswith("thrauto.npy")
    assert d == "ping_times_probe0_seg0_thr40_range1.npy"


def test_events_to_recording_time_with_offset_and_drift():
    log = np.arange(0, 200, 1.0)                    # Arduino clock pings
    true_slope, true_off = 1.0001, 12.5             # recording clock = slope*log + offset
    rec = true_slope * log + true_off
    ev = np.array([-5.0, 0.0, 33.3, 150.25, 199.0, 230.0])
    out = sync.events_to_recording_time(ev, log, rec)
    assert np.allclose(out, true_slope * ev + true_off, atol=1e-9)


def test_events_to_recording_time_piecewise():
    log = np.array([0.0, 1.0, 2.0, 3.0])
    rec = np.array([10.0, 11.0, 12.5, 13.5])        # a jump: local interpolation must follow it
    out = sync.events_to_recording_time([1.5], log, rec)
    assert out[0] == pytest.approx(11.75)
    with pytest.raises(ValueError):
        sync.events_to_recording_time([1.0], [0.0], [1.0])
    with pytest.raises(ValueError):
        sync.events_to_recording_time([1.0], [0.0, 0.0], [1.0, 2.0])


def test_match_ping_counts():
    log = np.arange(10.0)
    rec = np.r_[[3.0], 100.0 + np.arange(11.0)]     # one artefact ping at 3 s + one extra at the end
    with pytest.warns(UserWarning):
        lg, r = sync.match_ping_counts(log, rec, rec_min_time_s=50.0)
    assert lg.size == r.size == 10 and r[0] == 100.0
    with pytest.raises(ValueError):
        sync.match_ping_counts(log, rec, rec_min_time_s=50.0, on_mismatch="raise")
    fit = sync.ping_clock_fit(lg, r)
    assert fit["slope"] == pytest.approx(1.0) and fit["max_abs_residual_s"] < 1e-9
