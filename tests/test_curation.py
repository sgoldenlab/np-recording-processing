import numpy as np
import pandas as pd
import pytest

from np_ephys import curation as cu


def test_query_curate():
    m = pd.DataFrame({"amplitude_cutoff": [0.05, 0.2, 0.01], "isi_violations_ratio": [0.1, 0.1, 0.9],
                      "rp_contamination": [0.1, 0.1, 0.1]}, index=[10, 11, 12])
    assert list(cu.query_curate(m).index) == [10]


def test_bombcell_thresholds_are_the_lab_numbers():
    t = cu.BOMBCELL_THRESHOLDS
    assert t["noise"]["num_positive_peaks"] == {"greater": None, "less": 3}
    assert t["noise"]["peak_to_trough_duration"] == {"greater": 0.00005, "less": 0.00115}
    assert t["mua"]["snr"]["greater"] == 2.5 and t["mua"]["amplitude_median"]["abs"] is True
    assert t["mua"]["presence_ratio"] == {"greater": None, "less": None}
    assert t["mua"]["drift_ptp"]["less"] == 300
    assert t["non-somatic"]["trough_width"]["greater"] == 0.00015


def test_evaluate_thresholds():
    m = pd.DataFrame({"snr": [1.0, 3.0, 3.0], "amplitude_median": [-20.0, -10.0, -30.0],
                      "num_spikes": [500, 500, 100]})
    ok = cu.evaluate_thresholds(m, cu.BOMBCELL_THRESHOLDS["mua"])
    assert ok["snr"].tolist() == [False, True, True]
    assert ok["amplitude_median"].tolist() == [True, False, True]     # abs() > 15
    assert ok["num_spikes"].tolist() == [True, True, False]
    assert "presence_ratio" not in ok


def test_isi_stats():
    s = cu.isi_stats(np.arange(0, 1, 0.1))
    assert s["mean_isi_ms"] == pytest.approx(100.0) and s["cv_isi"] == pytest.approx(0.0, abs=1e-9)
    assert s["burst_fraction"] == 0.0
    assert cu.isi_stats([1.0])["mean_isi_ms"] is None
    burst = cu.isi_stats([0, 0.005, 0.010, 1.0])
    assert burst["burst_fraction"] == pytest.approx(2 / 3)


def test_classify_cell_type_ms_and_seconds():
    ms = pd.DataFrame({"peak_to_valley": [0.3, 0.4, 0.6, np.nan]})
    out = cu.classify_cell_type(ms)
    assert out["cell_type"].tolist() == ["putative_interneuron", "putative_interneuron",
                                        "putative_pyramidal", "unclassified"]
    sec = pd.DataFrame({"peak_to_valley": [0.0003, 0.0004, 0.0006, np.nan]})  # SpikeInterface-style seconds
    assert cu.classify_cell_type(sec)["cell_type"].tolist() == out["cell_type"].tolist()
    forced = cu.classify_cell_type(sec, ptv_unit="ms")      # explicit unit wins: everything narrow
    assert set(forced["cell_type"][:3]) == {"putative_interneuron"}
    assert out["ptv_threshold_ms"].iloc[0] == 0.4
