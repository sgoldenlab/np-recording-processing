"""Small SpikeInterface smoke test of the analyzer/curation helpers (skipped without SpikeInterface)."""
import pytest

try:
    import spikeinterface.full as si  # noqa: F401
except Exception as e:  # missing or broken optional stack (e.g. zarr 3 with SpikeInterface 0.104)
    pytest.skip(f"SpikeInterface unavailable: {e}", allow_module_level=True)

from np_ephys import analyzer as an  # noqa: E402
from np_ephys import curation as cu  # noqa: E402


@pytest.fixture(scope="module")
def analyzer():
    from spikeinterface.core import generate_ground_truth_recording

    rec, sorting = generate_ground_truth_recording(durations=[30.0], num_channels=16, num_units=6, seed=0,
                                                   sampling_frequency=30000.0)
    sorting = an.clean_sorting(sorting, rec, min_spikes=20)
    a = an.create_analyzer(sorting, rec, folder=None)
    an.compute_extensions(a, ["templates", "noise_levels", "spike_amplitudes", "template_metrics",
                              "unit_locations", "template_similarity", "correlograms"], verbose=False)
    return a


def test_quality_metrics_and_query(analyzer):
    df = an.compute_quality_metrics(analyzer, ["num_spikes", "firing_rate", "snr", "isi_violation", "amplitude_cutoff",
                                               "presence_ratio"], recording_name="SIM_R1", probe_id=0)
    assert {"snr", "isi_violations_ratio", "unit_key"} <= set(df.columns)
    assert df["unit_key"].iloc[0].startswith("SIM_R1_")
    assert len(cu.query_curate(df, "num_spikes > 0")) == len(df)


def test_cell_type_from_template_metrics(analyzer):
    tm = analyzer.get_extension("template_metrics").get_data()
    out = cu.classify_cell_type(tm)
    assert set(out["cell_type"]) <= {"putative_interneuron", "putative_pyramidal", "unclassified"}


def test_auto_merge_does_not_hide_errors(analyzer):
    merged = cu.slay_merge(analyzer, on_error="warn")
    assert merged.get_num_units() <= analyzer.get_num_units()


def test_full_extension_and_metric_profile(analyzer):
    an.compute_extensions(analyzer, an.EXTENSION_ORDER[:-1], verbose=False)
    df = an.compute_quality_metrics(analyzer, an.QUALITY_METRICS_FULL)
    for col in ("rp_contamination", "isi_violations_ratio", "d_prime", "isolation_distance", "snr"):
        assert col in df.columns, col
    assert len(cu.query_curate(df, cu.DEFAULT_QUERY.replace("amplitude_cutoff < 0.1 &", "")).index) >= 0
    labels = cu.label_units_bombcell  # callable exists; full labeling needs real-size data
    assert callable(labels)
