# np-recording-processing (`np_ephys`)

Neuropixels recording processing for the Golden Lab: compress SpikeGLX data, sort with Kilosort4 (via SpikeInterface), build analyzers and quality metrics, curate and classify units, and align spikes and behavioural events to the recording clock using Arduino/Raspberry Pi sync pings. `hardware/` holds the sync rig (sketch, Pi script, camera settings).

Repo URL: `https://github.com/sgoldenlab/np-recording-processing`. Lab protocols: see the lab protocol docs. Folder layout (lazy, configurable): [docs/folder_layout.md](docs/folder_layout.md).

## Install (uv)
```bash
uv venv --python 3.11 && source .venv/bin/activate   # Windows: .venv\Scripts\activate
uv pip install -e ".[dev]"            # light core: numpy, scipy, polars, pandas (sync/events/spiketimes/curation/layout)
uv pip install -e ".[compress]"       # + mtscomp
uv pip install -e ".[ephys]"          # + SpikeInterface 0.104.x, zarr<3 (loading, analyzer, curation)
# Sorting needs a CUDA GPU. Install the torch wheel for your driver FIRST, then:
uv pip install -e ".[sorting]"        # kilosort 4.1.3
```
`import np_ephys` works with only the core dependencies; heavy packages are imported inside functions.
SpikeInterface is pinned to 0.104.x because the API moved between releases.

## Pipeline
```
SpikeGLX *.ap.bin/.meta --01 compress--> raw/*.cbin,.ch,.meta
   --02 sort (KS4, per probe or per shank)--> processed/kilosort4_probe<N>/
   --03 preprocess + analyzer + metrics + curation--> processed/{preprocessed*.zarr, sorting_analyzer*.zarr, metrics/}
   --04 sync pings on the last (sync) channel--> derived/sync/ping_samples_*.npy (experiment window)
   --05 spikes in window, per-unit table--> derived/units/
   --06 Pi events.csv/frames.csv + pings--> events in video frames and recording time
```
Folders appear only when a step writes into them. One TOML config (`config/example_config.toml`, placeholder paths) drives `examples/01_...06_...`; every example also has `--demo` which runs on synthetic data.

## Modules
| Module / function | Goal | Key parameters (default) - what to change for a new project |
|---|---|---|
| `layout.SessionPaths`, `HistologyPaths`, `ensure_dir` | path arithmetic; folders are created only when a step writes | `[paths]` root, templates, names; session name `<animal>_R<n>` |
| `compress.compress_session` | `.ap.bin` -> mtscomp `.cbin` | `n_threads` = 80% of cores, `chunk_duration` 5 s; `concatenate` gives one sub-folder per segment |
| `io_raw.find_raw_files / load_recording` | locate and load (concatenated) AP files | `order` for explicit segment order; `include_sync` keeps the sync channel |
| `sorting.sort_session / build_ks_params` | Kilosort4 via `run_sorter` or `run_sorter_by_property` (by shank) | `batch_size` 15000, `nblocks` 1, `Th_universal` 9, `Th_learned` 8; `multiple_shanks` per probe |
| `gpu.*`, `sorting.configure_cuda_env` | free GPU memory between probes | `gpu_sleep_time` 10 s |
| `preprocess.kilosort_like_preprocess` | bandpass 300-6000 Hz, phase shift, bad-channel interpolation, spatial high-pass (per shank) | `freq_min/max` |
| `preprocess.make_lfp` | LFP 1-300 Hz, decimate x10 | `freq_min/max`, `decimate_factor` |
| `analyzer.clean_sorting / create_analyzer / compute_extensions` | analyzer with extensions in dependency order | `min_spikes` 500; random_spikes 300/unit seed 123; PCs 5; template similarity cosine |
| `analyzer.compute_quality_metrics` | lab metric set with lab parameters; fails loudly if a metric fails | `QUALITY_METRICS_FULL/QUICK`, `QUALITY_METRIC_PARAMS` |
| `analyzer.export_report / export_ibl_gui` | SpikeInterface report; IBL alignment GUI export | `lfp_band` (1, 300), `decimate_factor` 10 |
| `curation.query_curate` | threshold query | `DEFAULT_QUERY`: amplitude_cutoff < 0.1, isi_violations_ratio < 0.5, rp_contamination < 0.5 |
| `curation.slay_merge` | SLAy auto-merge | `on_error='raise'` |
| `curation.label_units_bombcell` | noise/mua/non-somatic/good | `BOMBCELL_THRESHOLDS` is data, edit or pass your own |
| `curation.classify_cell_type`, `isi_stats` | putative interneuron vs pyramidal | `ptv_threshold_ms` 0.4 |
| `sync.get_sync_edges / detect_rising_edges` | sync pings from the sync channel | `threshold` 40 (raw int16) or `None` for diff + find_peaks (0.5 / 0.5) |
| `sync.experiment_window`, `segment_sync_windows` | first ping to second-to-last ping; per-segment global windows | `end_index` -2 |
| `sync.events_to_recording_time` | event clock (Arduino) -> recording clock, piecewise linear between ping pairs | pairs from `match_ping_counts` (`rec_min_time_s`) |
| `events.*` | parse `EVENT_`, `TONESET_`, `Gas measurement` rows; frames to ping mapping | see `hardware/README.md` |
| `spiketimes.*` | spikes -> windowed per-unit table with firing rates | `windows` from `sync` |

## Quick start
```python
from np_ephys import sync, events as ev
from np_ephys.layout import SessionPaths
paths = SessionPaths.from_config(cfg, "NP01_R1")          # cfg = np_ephys.config.load_config("my.toml"); creates nothing
samples, times = sync.get_recording_sync(files, paths.sync_data, probe=0, threshold=40)
window = sync.experiment_window(samples)                   # (start_sample, end_sample)
table = ev.add_recording_time(ev.parse_event_labels(ev.load_events("X_events.csv")), times)
```

## CUDA watchdog fix (long Kilosort jobs aborted with CUDA launch-timeout errors)
Only relevant when an X display is attached to the GPU (headless VMs/containers have no watchdog). The NVIDIA driver can then abort long kernels. Add to the `Device` section of `/etc/X11/xorg.conf` (create with `sudo nvidia-xconfig` if missing), then reboot:
```
Section "Device"
    Identifier "Device0"
    Driver     "nvidia"
    Option     "Interactive" "0"
EndSection
```
`Interactive` is NVIDIA's documented option name. Also set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (`sorting.configure_cuda_env()` does this).

## Known caveats
* **Cell-type split** (`classify_cell_type`) uses a trough-to-peak threshold validated only for cortex and hippocampus. Do not use it for other regions (striatum, thalamus, brainstem...) without validating. SpikeInterface 0.104 reports durations in seconds; the function converts to ms.
* Sync pings are detected on the sync channel (last channel of the AP stream); ping k in the recording is log row k. If counts differ the function truncates at the end and warns; inspect `ping_clock_fit` residuals.
* Sync cache names encode probe, segment, range and threshold.
* `compute_quality_metrics` translates renamed metrics (isolation_distance -> `mahalanobis`) and drops parameters the installed SpikeInterface does not accept (with a warning).
* GPL tools (Kilosort, bombcell) are only called, never copied.

## Credits & citations
Please cite what you use (full entries in `CITATION.cff`):
* SpikeInterface: Buccino et al., "SpikeInterface, a unified framework for spike sorting", eLife 9:e61834 (2020). MIT.
* Kilosort4: Pachitariu, Sridhar, Pennington, Stringer, "Spike sorting with Kilosort4", Nature Methods 21, 914-921 (2024). GPL-3.0 (wrapped).
* Bombcell: Fabre, van Beest, Peters, Carandini, Harris, Zenodo, doi:10.5281/zenodo.8172821 (2023). GPL-3.0 (thresholds follow its categories; implementation via SpikeInterface).
* mtscomp (International Brain Laboratory): https://github.com/int-brain-lab/mtscomp, BSD-3-Clause. Cite the GitHub repository.
* SLAy merge preset: Koukuntla et al., 2025 <!-- TODO verify -->.
* IBL alignment GUI / export: [iblapps](https://github.com/int-brain-lab/iblapps) (MIT).
* np-ephys: Kevin N. Schneider (BSD-3-Clause).

License: BSD-3-Clause (see `LICENSE`).

## Status
Verified in a sandbox: `py_compile`, `pytest` (53 passed: synthetic arrays for sync, events, spiketimes, curation, layout, compression; SpikeInterface 0.104.0 smoke tests on a ground-truth recording), `ruff`, and every example in `--demo` mode.
Not verified: Kilosort4 sorting (needs a GPU), real Neuropixels data and real `.cbin` loading, `kilosort_like_preprocess` on real probes, IBL GUI export, bombcell labelling on realistically sized data, `get_recording_sync` on real files, the hardware scripts (need Pi/Arduino/AA-8000), SpikeInterface other than 0.104.0, Windows.
