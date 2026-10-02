"""Kilosort4 spike sorting through SpikeInterface (single shank or per shank).

Kilosort4 does its own high-pass filtering, common-average referencing, whitening and drift
correction, so the recording passed here is the RAW (compressed) AP stream without SpikeInterface
preprocessing. Kilosort/torch/SpikeInterface are imported lazily.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .io_raw import find_raw_files, load_recording
from .layout import SessionPaths, ensure_dir

#: Kilosort4 overrides actually used by the lab (everything else = SpikeInterface/KS4 defaults).
#: batch_size=15000, nblocks=1 is a memory-safe set for a 16 GB GPU.
DEFAULT_KS_OVERRIDES: dict[str, Any] = dict(
    batch_size=15000,
    nblocks=1,  # lowering from 3-5 greatly reduces memory in drift correction
    Th_universal=9.0,
    Th_learned=8.0,
    save_preprocessed_copy=False,
    save_extra_vars=False,
)
DEFAULT_JOB_KWARGS: dict[str, Any] = dict(n_jobs=4, chunk_duration="1s", progress_bar=True)


class SortingExecutionError(RuntimeError):
    """Kilosort run failed."""


def configure_cuda_env(expandable_segments: bool = True, launch_blocking: bool = False) -> None:
    """Set CUDA/PyTorch env vars. Call explicitly (before importing torch) - not done at import."""
    if expandable_segments:
        os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    if launch_blocking:
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # synchronous kernels: better tracebacks, slower


def build_ks_params(overrides: dict | None = None, base: dict | None = None) -> dict:
    """SpikeInterface's Kilosort4 defaults + :data:`DEFAULT_KS_OVERRIDES` + ``overrides``.

    ``base`` replaces the SpikeInterface defaults (useful offline/in tests).
    """
    if base is None:
        import spikeinterface.full as si

        base = si.get_default_sorter_params("kilosort4")
    return {**base, **DEFAULT_KS_OVERRIDES, **(overrides or {})}


def prepare_probe_configuration(multiple_shanks, n_probes: int) -> list[bool]:
    """Per-probe multi-shank flags: ``True`` -> ``[True]*n``; short lists are padded with False."""
    flags = list(multiple_shanks) if isinstance(multiple_shanks, (list, tuple)) else [bool(multiple_shanks)] * n_probes
    return (flags + [False] * n_probes)[:n_probes]


def run_single_shank_sorting(recording, output_folder: Path, ks_params: dict, overwrite: bool = False,
                             save_sorting: bool = True):
    """``si.run_sorter('kilosort4', ...)``; saves ``{output_folder}/sorting.zarr``."""
    import spikeinterface.full as si

    output_folder = Path(output_folder)
    if output_folder.exists() and not overwrite:
        print(f"skipping existing sorting: {output_folder}")
        return None
    try:
        sorting = si.run_sorter("kilosort4", recording, folder=output_folder, remove_existing_folder=True,
                                docker_image=False, verbose=True, **ks_params)
        if save_sorting:
            sorting.save_to_zarr(output_folder / "sorting.zarr", overwrite=True)
        return sorting
    except Exception as e:  # noqa: BLE001
        raise SortingExecutionError(f"single-shank sorting failed: {e}") from e


def run_multi_shank_sorting(recording, output_folder: Path, ks_params: dict, overwrite: bool = False,
                            save_aggregated: bool = True):
    """Sort every shank (recording property ``group``) separately and aggregate.

    Saves ``{output_folder}/aggregated_sorting.zarr``.
    """
    import spikeinterface.full as si

    output_folder = Path(output_folder)
    if output_folder.exists() and not overwrite:
        print(f"skipping existing sorting: {output_folder}")
        return None
    try:
        sorting = si.run_sorter_by_property("kilosort4", recording, grouping_property="group",
                                            folder=output_folder, remove_existing_folder=True,
                                            docker_image=False, **ks_params)
        if save_aggregated:
            sorting.save_to_zarr(output_folder / "aggregated_sorting.zarr", overwrite=True)
        return sorting
    except Exception as e:  # noqa: BLE001
        raise SortingExecutionError(f"multi-shank sorting failed: {e}") from e


def sort_probe(recording, output_folder: Path, ks_params: dict, multiple_shanks: bool = False,
               overwrite: bool = False, channel_map_png: Path | None = None):
    """Sort one probe (and optionally save the probe layout plot). Returns the sorting or None."""
    if channel_map_png is not None:
        save_probe_channel_map(recording, channel_map_png)
    fn = run_multi_shank_sorting if multiple_shanks else run_single_shank_sorting
    return fn(recording, output_folder, ks_params, overwrite=overwrite)


def save_probe_channel_map(recording, png_path: Path, figsize=(10, 5), dpi: int = 400) -> Path | None:
    """Plot the probe contacts used (sanity check of channel selection / shanks)."""
    import matplotlib.pyplot as plt
    from probeinterface.plotting import plot_probe

    try:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        plot_probe(recording.get_probe(), ax=ax)
        Path(png_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return Path(png_path)
    except Exception as e:  # noqa: BLE001
        print(f"(!) could not save probe map: {e}")
        return None


def sort_session(paths: SessionPaths, properties: dict, ks_params: dict, overwrite: bool = False,
                 gpu_sleep_time: float = 10) -> dict:
    """Sort all probes of one session. Output: ``<session>/processed/kilosort4_probe{imec}/`` (created when sorting starts).

    ``properties``: ``{'concatenate': bool, 'multiple_shanks': bool | list[bool]}``.
    Returns ``{'session', 'status', 'probes_processed', 'errors'}``; status is
    ``success | partial | failed | skipped`` (probes with existing output and ``overwrite=False`` count as done). One failing probe does not stop the others.
    """
    from .gpu import gpu_memory_context

    res = {"session": paths.session, "status": "unknown", "probes_processed": [], "skipped_probes": [],
           "errors": []}
    concatenate = properties.get("concatenate", False)
    if not paths.raw.exists():
        res.update(status="skipped", errors=[f"missing {paths.raw}"])
        return res
    raw = find_raw_files(paths.raw, paths.session, concatenate=concatenate)
    if not raw:
        res.update(status="skipped", errors=["no raw files"])
        return res
    shanks = prepare_probe_configuration(properties.get("multiple_shanks", False), len(raw))
    for pos, (probe, files) in enumerate(raw.items()):
        try:
            ensure_dir(paths.processed)
            with gpu_memory_context(gpu_sleep_time):
                rec = load_recording(files, include_sync=False, concatenate=concatenate)
                sorting = sort_probe(rec, paths.kilosort_dir(probe), ks_params, multiple_shanks=shanks[pos],
                                     overwrite=overwrite,
                                     channel_map_png=paths.processed / f"{paths.session}_probe{probe}_channel_selection.png")
            if sorting is None:  # existing output and overwrite=False
                res["skipped_probes"].append(probe)
            else:
                res["probes_processed"].append(probe)
        except Exception as e:  # noqa: BLE001
            res["errors"].append(f"probe {probe}: {e}")
    n_ok = len(res["probes_processed"]) + len(res["skipped_probes"])
    res["status"] = "success" if n_ok == len(raw) else ("partial" if n_ok else "failed")
    return res
