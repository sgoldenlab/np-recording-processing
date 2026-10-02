"""SortingAnalyzer helpers: clean sorting, create analyzer, compute extensions in dependency order,
quality metrics, reports and IBL-GUI export (SpikeInterface, lazy imports).

Extension *order matters*: random_spikes -> waveforms -> templates -> (noise_levels, spike_amplitudes,
principal_components) -> (template_similarity, template_metrics, unit_locations) -> quality_metrics.
"""
from __future__ import annotations

import inspect
import warnings
from pathlib import Path

#: canonical computation order
EXTENSION_ORDER = [
    "random_spikes", "waveforms", "templates", "noise_levels", "isi_histograms", "correlograms",
    "spike_amplitudes", "principal_components", "template_similarity", "template_metrics",
    "unit_locations", "quality_metrics",
]
EXTENSION_REQUIRES = {
    "waveforms": ["random_spikes"],
    "templates": ["random_spikes", "waveforms"],
    "spike_amplitudes": ["templates"],
    "principal_components": ["random_spikes", "waveforms"],
    "template_similarity": ["templates"],
    "template_metrics": ["templates"],
    "unit_locations": ["templates"],
    "quality_metrics": ["noise_levels", "spike_amplitudes", "principal_components"],
}
#: parameters used by the lab
EXTENSION_PARAMS = {
    "random_spikes": dict(max_spikes_per_unit=300, seed=123),
    "principal_components": dict(n_components=5, mode="by_channel_local"),
    "template_similarity": dict(method="cosine_similarity"),
    "unit_locations": dict(method="grid_convolution"),
}
QUALITY_METRICS_FULL = [
    "num_spikes", "firing_rate", "amplitude_cutoff", "presence_ratio", "snr",
    "isi_violation", "rp_violation", "d_prime", "isolation_distance",
]
QUALITY_METRICS_QUICK = ["num_spikes", "firing_rate", "snr", "isi_violation", "amplitude_cutoff", "presence_ratio"]
QUALITY_METRIC_PARAMS = {
    "num_spikes": {},
    "firing_rate": {},
    "presence_ratio": {"bin_duration_s": 180, "mean_fr_ratio_thresh": 0.0},
    "snr": {"peak_sign": "neg", "peak_mode": "extremum"},
    "isi_violation": {"isi_threshold_ms": 1.0, "min_isi_ms": 0},
    "rp_violation": {"refractory_period_ms": 1.0, "censored_period_ms": 0.0},
    "amplitude_cutoff": {"peak_sign": "neg", "num_histogram_bins": 100, "histogram_smoothing_value": 3,
                         "amplitudes_bins_min_ratio": 5},
    "isolation_distance": {},
    "d_prime": {},
}
#: extensions needed by the quick profile (no PCs / spike amplitudes metrics)
QUICK_EXTENSIONS = ["random_spikes", "waveforms", "templates", "noise_levels", "unit_locations",
                    "template_metrics", "quality_metrics"]


def order_extensions(requested, add_dependencies: bool = True) -> list[str]:
    """Sort extension names into a valid computation order (adds missing prerequisites)."""
    unknown = set(requested) - set(EXTENSION_ORDER)
    if unknown:
        raise ValueError(f"unknown extensions: {sorted(unknown)}")
    wanted = set(requested)
    if add_dependencies:
        stack = list(wanted)
        while stack:
            for dep in EXTENSION_REQUIRES.get(stack.pop(), []):
                if dep not in wanted:
                    wanted.add(dep)
                    stack.append(dep)
    return [e for e in EXTENSION_ORDER if e in wanted]


def clean_sorting(sorting, recording, min_spikes: int = 500, remove_redundant: bool = True):
    """Register recording, drop spikes beyond the recording, drop redundant units, keep >= ``min_spikes``.

    500 spikes is the minimum for ``random_spikes`` (300/unit) to be meaningful. A
    "unit_ids dtype" ValueError of some SpikeInterface versions in ``remove_redundant_units`` is
    caught with a warning (the step is then skipped); any other error is raised.
    """
    import spikeinterface.full as si

    sorting.register_recording(recording)
    sorting = si.remove_excess_spikes(sorting, recording)
    if remove_redundant:
        try:
            sorting = si.remove_redundant_units(sorting, align=False, remove_strategy="max_spikes")
        except ValueError as e:
            if "unit_ids dtype" not in str(e):
                raise
            warnings.warn(f"remove_redundant_units skipped (SpikeInterface dtype issue): {e}")
    counts = sorting.count_num_spikes_per_unit(outputs="array")
    return sorting.select_units(sorting.get_unit_ids()[counts >= min_spikes])


def create_analyzer(sorting, recording, folder: Path | None = None, format: str = "zarr", sparse: bool = True,
                    compress: bool = True, overwrite: bool = True):
    """Create a (sparse) SortingAnalyzer; zarr with LZMA by default (``folder=None`` -> in memory)."""
    import spikeinterface.full as si

    kw = {}
    if folder is None:
        format = "memory"
    elif format == "zarr" and compress:
        from numcodecs import LZMA

        kw["backend_options"] = dict(compressor=LZMA())
    return si.create_sorting_analyzer(sorting=sorting, recording=recording,
                                      folder=None if folder is None else str(folder), format=format,
                                      sparse=sparse, overwrite=overwrite, **kw)


def compute_extensions(analyzer, which=EXTENSION_ORDER, params: dict | None = None, recompute: bool = False,
                       verbose: bool = True):
    """Compute ``which`` extensions in dependency order, skipping existing ones unless ``recompute``.

    ``quality_metrics`` is skipped here (use :func:`compute_quality_metrics`, which takes metric names/params).
    """
    p = {**EXTENSION_PARAMS, **(params or {})}
    for name in order_extensions([w for w in which if w != "quality_metrics"]):
        if analyzer.has_extension(name) and not recompute:
            continue
        analyzer.compute(name, **p.get(name, {}), verbose=verbose)
    return analyzer


def _metric_functions() -> dict:
    """Map our metric names to SpikeInterface functions (module moved between versions)."""
    try:
        from spikeinterface.metrics.quality import misc_metrics as mm  # >= 0.104
    except ImportError:  # pragma: no cover - 0.102 / 0.103
        from spikeinterface.qualitymetrics import misc_metrics as mm
    names = {"amplitude_cutoff": ("compute_amplitude_cutoffs",), "snr": ("compute_snrs",),
             "isi_violation": ("compute_isi_violations",), "presence_ratio": ("compute_presence_ratios",),
             "rp_violation": ("compute_refrac_period_violations", "compute_rp_violations")}
    out = {}
    for k, cands in names.items():
        for c in cands:
            if hasattr(mm, c):
                out[k] = getattr(mm, c)
    return out


def adapt_metric_params(metric_names, params: dict) -> dict:
    """Drop parameters the installed SpikeInterface no longer accepts (with a warning).

    Example: ``amplitude_cutoff(peak_sign=...)`` exists in SpikeInterface 0.102 but not in >= 0.104,
    where passing it makes the metric fail (SpikeInterface only warns). Lab values are kept whenever accepted.
    """
    funcs = _metric_functions()
    out = {}
    for name in metric_names:
        p = dict(params.get(name, {}))
        fn = funcs.get(name)
        if fn is not None:
            ok = set(inspect.signature(fn).parameters)
            dropped = [k for k in p if k not in ok]
            if dropped:
                warnings.warn(f"metric '{name}': parameters {dropped} not supported by this SpikeInterface; dropped")
                p = {k: v for k, v in p.items() if k in ok}
        out[name] = p
    return out


# Metrics renamed/reorganised in newer SpikeInterface: old name -> new computation name
# (the output columns keep the old names, e.g. 'mahalanobis' yields isolation_distance and l_ratio).
RENAMED_METRICS = {"isolation_distance": "mahalanobis", "l_ratio": "mahalanobis",
                   "nn_isolation": "nn_advanced", "nn_noise_overlap": "nn_advanced"}


def translate_metric_names(metric_names, available) -> list[str]:
    """Map lab metric names to the installed SpikeInterface's computation names (deduplicated, ordered)."""
    available = set(available)
    out: list[str] = []
    for n in metric_names:
        new = n if n in available or n not in RENAMED_METRICS else RENAMED_METRICS[n]
        if new not in out:
            out.append(new)
    return out


def compute_quality_metrics(analyzer, metric_names=QUALITY_METRICS_FULL, metric_params: dict | None = None,
                            recording_name: str | None = None, probe_id: int | None = None, strict: bool = True):
    """Quality metrics as a pandas DataFrame indexed by unit id (extra ``unit_key`` etc. if names given).

    ``strict=True`` raises if SpikeInterface reports a failed metric.
    Needs noise_levels, and spike_amplitudes / principal_components for amplitude_cutoff, d_prime,
    isolation_distance (see :func:`compute_extensions`).
    """
    params = adapt_metric_params(metric_names, {**QUALITY_METRIC_PARAMS, **(metric_params or {})})
    try:
        from spikeinterface.metrics.quality import get_quality_metric_list  # SI >= 0.104
        available = list(get_quality_metric_list())
    except Exception:
        available = list(metric_names)  # older SpikeInterface: names unchanged
    metric_names = translate_metric_names(metric_names, available)
    params = {k: v for k, v in params.items() if k in metric_names}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ext = analyzer.compute("quality_metrics", metric_names=list(metric_names), metric_params=params,
                               verbose=True)
    failed = [str(w.message) for w in caught if "Error computing metric" in str(w.message)]
    for w in caught:  # re-emit everything else
        if "Error computing metric" not in str(w.message):
            warnings.warn_explicit(w.message, w.category, w.filename, w.lineno)
    if failed:
        # SpikeInterface only warns when a metric fails, leaving the column missing/NaN.
        if strict:
            raise RuntimeError("quality metric(s) failed: " + "; ".join(failed))
        warnings.warn("quality metric(s) failed: " + "; ".join(failed))
    df = ext.get_data().copy()
    if recording_name is not None:
        from .spiketimes import make_unit_key

        df["unit_key"] = [make_unit_key(recording_name, u, probe_id) for u in df.index]
        df["recording_name"] = recording_name
    if probe_id is not None:
        df["probe_id"] = probe_id
    return df


def export_report(analyzer, folder: Path, remove_if_exists: bool = True):
    """HTML-like summary report folder (``si.export_report``)."""
    import spikeinterface.full as si

    si.export_report(analyzer, output_folder=folder, remove_if_exists=remove_if_exists)


def export_ibl_gui(analyzer, folder: Path, lfp_band=(1.0, 300.0), decimate_factor: int = 10,
                   compute_lfp: bool = True, remove_if_exists: bool = True, **kwargs):
    """Export for the IBL ephys alignment GUI (``si.export_to_ibl_gui``).

    Computes ``amplitude_median`` first (needed by the GUI) and, if ``compute_lfp``, passes an LFP
    recording (bandpass 1-300 Hz, decimated by 10) so the GUI can show LFP power.
    """
    import spikeinterface.full as si

    analyzer.compute("quality_metrics", metric_names=["amplitude_median"])
    lfp = None
    if compute_lfp:
        lfp = si.decimate(si.bandpass_filter(analyzer.recording, freq_min=lfp_band[0], freq_max=lfp_band[1]),
                          decimate_factor)
    si.export_to_ibl_gui(sorting_analyzer=analyzer, output_folder=folder, lfp_recording=lfp,
                         remove_if_exists=remove_if_exists, verbose=True, **kwargs)
