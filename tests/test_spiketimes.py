import polars as pl
import pytest

from np_ephys import spiketimes as st


def test_unit_key():
    assert st.make_unit_key("NP09_R1", 5) == "NP09_R1_5"
    assert st.make_unit_key("NP09_R1", 5, 1, 2) == "NP09_R1_p1_s2_u5"
    assert st.make_unit_key("NP09_R1", 5, 0) == "NP09_R1_p0_u5"


def test_spikes_to_frame_uses_ids_not_indices():
    # units were curated: ids are NOT 0..N-1
    df = st.spikes_to_frame([10, 20, 30], [0, 2, 1], unit_ids=[7, 42, 99])
    assert df["unit_id"].to_list() == [7, 99, 42]


def _spikes():
    return pl.DataFrame({"sample_index": [100, 200, 300, 1000, 2000, 5000, 6000], "unit_id": [1, 1, 2, 1, 2, 2, 1]})


def test_filter_windows():
    s = _spikes()
    assert st.filter_spikes_by_windows(s, (200, 1000)).height == 3
    wins = [{"start_sync": 100, "end_sync": 200}, {"start_sync": 5000, "end_sync": 6000}]
    assert st.filter_spikes_by_windows(s, wins)["sample_index"].to_list() == [100, 200, 5000, 6000]
    assert st.filter_spikes_by_windows(s, []).height == s.height
    assert st.windows_duration_samples(wins) == 1100


def test_aggregate_firing_rate():
    s = _spikes()
    agg = st.aggregate_spikes_per_unit(s, 1000.0, total_duration_samples=10_000)  # 10 s
    r1 = agg.filter(pl.col("unit_id") == 1).row(0, named=True)
    assert r1["num_spikes"] == 4 and r1["firing_rate"] == pytest.approx(0.4)
    assert r1["spike_times_sec"] == pytest.approx([0.1, 0.2, 1.0, 6.0])
    assert r1["spike_times"] == [100, 200, 1000, 6000]
    # default duration = span
    agg2 = st.aggregate_spikes_per_unit(s, 1000.0)
    assert agg2.filter(pl.col("unit_id") == 2)["firing_rate"][0] == pytest.approx(3 / 5.9)


def test_build_unit_table_with_windows_and_metadata():
    md = pl.DataFrame({"unit_id": [1, 2, 3], "brain_region": ["a", "b", "c"]})
    out = st.build_unit_table(_spikes(), md, 1000.0, windows=[(0, 1000), (5000, 6000)])
    assert out["unit_id"].to_list() == [1, 2]            # unit 3 has no spikes -> dropped
    u1 = out.filter(pl.col("unit_id") == 1).row(0, named=True)
    assert u1["num_spikes"] == 4 and u1["firing_rate"] == pytest.approx(4 / 2.0)   # 2 s total window


def test_segment_windows_integration():
    from np_ephys import sync
    b = sync.segment_boundaries_from_lengths([10_000, 10_000], 1000.0)
    wins = sync.segment_sync_windows(b, [[500, 600, 9000, 9500], [100, 200, 8000, 9000]])
    spikes = pl.DataFrame({"sample_index": [400, 700, 9300, 10_050, 10_150, 18_500, 19_500],
                           "unit_id": [1] * 7})
    kept = st.filter_spikes_by_windows(spikes, wins)["sample_index"].to_list()
    assert kept == [700, 10_150]
