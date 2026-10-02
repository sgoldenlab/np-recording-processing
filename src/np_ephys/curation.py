"""Unit curation and classification.

* :func:`query_curate`  - lab default threshold query on quality metrics.
* :func:`slay_merge` / :func:`auto_merge`  - SpikeInterface auto-merge (SLAy preset) that does NOT hide failures.
* :data:`BOMBCELL_THRESHOLDS` + :func:`label_units_bombcell` - noise / mua / non-somatic / good labels
  (wraps SpikeInterface's bombcell implementation; bombcell itself is GPL-3.0, so no code is copied).
* :func:`classify_cell_type`, :func:`isi_stats` - putative interneuron / pyramidal by trough-to-peak duration.
"""
from __future__ import annotations

import copy
import inspect
import warnings

import numpy as np

#: lab default auto-curation (applied after the >= 500 spike filter); intentionally lenient
DEFAULT_QUERY = "amplitude_cutoff < 0.1 & isi_violations_ratio < 0.5 & rp_contamination < 0.5"

#: Lab bombcell thresholds ({'greater': lower bound, 'less': upper bound}; None = unused;
#: 'abs': compare absolute value). Comments give the bombcell default where relaxed.
#: Durations in seconds. Metric names follow SpikeInterface (``rp_contamination`` needs the 'rp_violation'
#: quality metric; SpikeInterface >= 0.105 defaults use 'sliding_rp_violation' instead).
BOMBCELL_THRESHOLDS = {
    "noise": {
        "num_positive_peaks": {"greater": None, "less": 3},
        "num_negative_peaks": {"greater": None, "less": 1},
        "peak_to_trough_duration": {"greater": 0.00005, "less": 0.00115},
        "waveform_baseline_flatness": {"greater": None, "less": 0.5},
        "peak_after_to_trough_ratio": {"greater": None, "less": 1.5},
        "exp_decay": {"greater": 0.01, "less": 0.1},
    },
    "mua": {
        "amplitude_median": {"greater": 15, "less": None, "abs": True},  # lowered from 30
        "snr": {"greater": 2.5, "less": None},                           # lowered from 5
        "amplitude_cutoff": {"greater": None, "less": 0.5},              # relaxed from 0.2
        "num_spikes": {"greater": 300, "less": None},
        "rp_contamination": {"greater": None, "less": 0.5},              # relaxed from 0.1
        "presence_ratio": {"greater": None, "less": None},               # disabled (default 0.7)
        "drift_ptp": {"greater": None, "less": 300},                     # relaxed from 100
    },
    "non-somatic": {
        "peak_before_to_trough_ratio": {"greater": None, "less": 5},
        "peak_before_width": {"greater": 0.0001, "less": None},
        "trough_width": {"greater": 0.00015, "less": None},
        "peak_before_to_peak_after_ratio": {"greater": None, "less": 3},
        "main_peak_to_trough_ratio": {"greater": None, "less": 0.8},
    },
}


def query_curate(metrics, query: str = DEFAULT_QUERY):
    """Rows (units) of a metrics DataFrame that pass ``query`` (pandas ``DataFrame.query`` syntax)."""
    return metrics.query(query)


def auto_merge(analyzer, preset: str = "slay", on_error: str = "raise", **kwargs):
    """``si.auto_merge_units(analyzer, presets=preset, recursive=True, merging_mode='hard')``.

    ``on_error='raise'`` (default) propagates the error; ``'warn'`` warns and returns a copy of
    the input analyzer. Requires the 'template_similarity' and 'correlograms' extensions (and
    'random_spikes'/'templates') - see :func:`np_ephys.analyzer.compute_extensions`.
    """
    import spikeinterface.full as si

    opts = dict(presets=preset, recursive=True, merging_mode="hard")
    if "raise_error" in inspect.signature(si.auto_merge_units).parameters:
        opts["raise_error"] = on_error == "raise"
    opts.update(kwargs)
    try:
        return si.auto_merge_units(analyzer, **opts)
    except Exception as e:  # noqa: BLE001
        if on_error == "raise":
            raise
        warnings.warn(f"auto-merge failed ({type(e).__name__}: {e}); continuing WITHOUT merges")
        return analyzer.copy()


def slay_merge(analyzer, on_error: str = "raise", **kwargs):
    """Auto-merge with the SLAy preset (Koukuntla et al. 2025; see README)."""
    return auto_merge(analyzer, preset="slay", on_error=on_error, **kwargs)


def remove_redundant_and_duplicates(analyzer):
    """Pre-merge cleaning: redundant units (align, minimum_shift) and duplicated spikes.

    Returns the analyzer restricted to the surviving units. The 'unit_ids dtype' SpikeInterface
    ValueError is caught with a warning; everything else is raised.
    """
    import spikeinterface.full as si

    sorting = analyzer.sorting
    try:
        sorting = si.remove_redundant_units(analyzer, align=True, remove_strategy="minimum_shift")
    except ValueError as e:
        if "unit_ids dtype" not in str(e):
            raise
        warnings.warn(f"remove_redundant_units skipped: {e}")
    sorting = si.remove_duplicated_spikes(sorting, method="keep_first_iterative")
    return analyzer.select_units(sorting.unit_ids)


def label_units_bombcell(analyzer, thresholds: dict | None = None):
    """Label units 'noise' | 'mua' | 'non-somatic' | 'good' (pandas Series indexed by unit id).

    Needs SpikeInterface >= 0.104 and the extensions spike_amplitudes, amplitude_scalings,
    spike_locations, noise_levels, template_metrics, drift metrics and quality_metrics.
    """
    from spikeinterface.curation import bombcell_label_units

    return bombcell_label_units(sorting_analyzer=analyzer,
                                thresholds=copy.deepcopy(thresholds or BOMBCELL_THRESHOLDS))


def evaluate_thresholds(metrics, group_thresholds: dict):
    """Boolean pass/fail table (units x metrics) for one threshold group, for INSPECTION.

    ``True`` = the unit satisfies the criterion (``greater < value < less``). Metrics missing from
    ``metrics`` are skipped. This does not reproduce bombcell's label precedence; use
    :func:`label_units_bombcell` for labels.
    """
    import pandas as pd

    out = {}
    for name, th in group_thresholds.items():
        if name not in metrics.columns or (th.get("greater") is None and th.get("less") is None):
            continue
        v = metrics[name].abs() if th.get("abs") else metrics[name]
        ok = pd.Series(True, index=metrics.index)
        if th.get("greater") is not None:
            ok &= v > th["greater"]
        if th.get("less") is not None:
            ok &= v < th["less"]
        out[name] = ok
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------- cell types

def isi_stats(spike_times_sec) -> dict:
    """ISI statistics of one unit: mean/median ISI (ms), CV of ISI, fraction of ISIs < 10 ms."""
    st = np.sort(np.asarray(spike_times_sec, dtype=float))
    if st.size < 2:
        return dict(mean_isi_ms=None, median_isi_ms=None, cv_isi=None, burst_fraction=None)
    isi = np.diff(st) * 1000.0
    mean = float(isi.mean())
    return dict(mean_isi_ms=mean, median_isi_ms=float(np.median(isi)),
                cv_isi=float(isi.std() / mean) if mean > 0 else None,
                burst_fraction=float(np.mean(isi < 10.0)))


def classify_cell_type(template_metrics, ptv_threshold_ms: float = 0.4, ptv_column: str | None = None,
                       ptv_unit: str = "auto"):
    """Putative cell type from trough-to-peak duration (Guo et al. 2025, Anesthesiology, Fig. 3).

    ``ptv_column`` defaults to ``peak_to_valley`` (SpikeInterface 0.102) or ``peak_to_trough_duration`` (>= 0.103/0.104).
    ``<= 0.4 ms`` -> ``putative_interneuron`` (narrow), ``> 0.4 ms`` -> ``putative_pyramidal``,
    missing -> ``unclassified``. Adds ``cell_type`` and ``ptv_threshold_ms`` columns.

    ``ptv_unit``: 'ms', 's', or 'auto' (SpikeInterface template metrics are in SECONDS in recent
    versions; 'auto' treats a median < 0.1 as seconds). CAVEAT: a single waveform-width cut-off is
    only reliable for cortex / hippocampus; subcortical regions (thalamus, cerebellum, ...) do not
    follow the convention.
    """
    import pandas as pd

    df = template_metrics.copy()
    if ptv_column is None:  # column was renamed between SpikeInterface versions
        ptv_column = next((c for c in ("peak_to_valley", "peak_to_trough_duration") if c in df.columns), None)
        if ptv_column is None:
            raise KeyError("no 'peak_to_valley' / 'peak_to_trough_duration' column in template metrics")
    ptv = pd.to_numeric(df[ptv_column], errors="coerce")
    if ptv_unit == "auto":
        med = ptv.dropna().median()
        ptv_unit = "s" if (pd.notna(med) and med < 0.1) else "ms"
    ptv_ms = ptv * 1000.0 if ptv_unit == "s" else ptv
    cell = np.where(ptv_ms.isna(), "unclassified",
                    np.where(ptv_ms <= ptv_threshold_ms, "putative_interneuron", "putative_pyramidal"))
    df["cell_type"] = cell
    df["ptv_ms"] = ptv_ms
    df["ptv_threshold_ms"] = ptv_threshold_ms
    return df
