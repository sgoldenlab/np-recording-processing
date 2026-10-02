import pytest

from np_ephys import analyzer as an
from np_ephys.sorting import DEFAULT_KS_OVERRIDES, build_ks_params, prepare_probe_configuration


def test_order_adds_dependencies_and_sorts():
    assert an.order_extensions(["templates"]) == ["random_spikes", "waveforms", "templates"]
    out = an.order_extensions(["template_metrics", "noise_levels", "random_spikes"])
    assert out == ["random_spikes", "waveforms", "templates", "noise_levels", "template_metrics"]
    assert an.order_extensions(["quality_metrics"])[-1] == "quality_metrics"
    assert an.order_extensions(["templates"], add_dependencies=False) == ["templates"]
    with pytest.raises(ValueError):
        an.order_extensions(["bogus"])


def test_full_order_is_dependency_consistent():
    pos = {e: i for i, e in enumerate(an.EXTENSION_ORDER)}
    for e, deps in an.EXTENSION_REQUIRES.items():
        assert all(pos[d] < pos[e] for d in deps), e


def test_lab_parameters():
    assert an.EXTENSION_PARAMS["random_spikes"] == dict(max_spikes_per_unit=300, seed=123)
    assert an.EXTENSION_PARAMS["principal_components"] == dict(n_components=5, mode="by_channel_local")
    qp = an.QUALITY_METRIC_PARAMS
    assert qp["presence_ratio"] == {"bin_duration_s": 180, "mean_fr_ratio_thresh": 0.0}
    assert qp["amplitude_cutoff"]["histogram_smoothing_value"] == 3
    assert qp["rp_violation"]["refractory_period_ms"] == 1.0
    assert set(an.QUALITY_METRICS_FULL) >= {"d_prime", "isolation_distance", "rp_violation"}


def test_ks_params_and_shank_config():
    p = build_ks_params({"Th_universal": 7.0}, base={"batch_size": 60000, "x": 1})
    assert p["batch_size"] == 15000 and p["nblocks"] == 1 and p["Th_universal"] == 7.0
    assert p["Th_learned"] == 8.0 and p["x"] == 1
    assert DEFAULT_KS_OVERRIDES["save_preprocessed_copy"] is False
    assert prepare_probe_configuration(True, 2) == [True, True]
    assert prepare_probe_configuration([True], 2) == [True, False]
    assert prepare_probe_configuration(False, 1) == [False]
