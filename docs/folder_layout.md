# Folder layout (lazy and configurable)

Nothing is created in advance. `np_ephys.layout` only *resolves* paths; the step that writes an output creates
the folder it needs (`ensure_dir` / `ensure_parent`). A session without LFP or alignment data therefore never
gets an empty `alignment/` or `lfp/` folder, and a folder that exists tells you a step actually produced something.

```
<root>/<animal>/
  histology/                       HistologyPaths   (animal id = text before the first "_" of the session)
    downsampled/                   Arivis output (rotated, not registered)
    registered/                    brainreg output
    stacks/                        optional ImageJ outputs (trimmed / reoriented / NRRD stacks)
    tracks/                        Lasagna *_pts.csv
  <animal>_R<n>/                   SessionPaths     (one recording session)
    raw/                           .cbin/.ch/.meta (concatenated sessions: one sub-folder per SpikeGLX run)
    processed/                     kilosort4_probe<N>/, preprocessed_recording_probe<N>.zarr,
                                   sorting_analyzer[_clean|_quick|_slay]_probe<N>.zarr, lfp/, metrics/
    alignment/                     ibl_alignment_SI_probe<N>/ (IBL GUI export), channel tables, resolved regions
    derived/                       sync/ (ping_samples_*.npy caches), units/ (per-unit tables in the experiment window)
```

Pipeline order is given by the READMEs and protocols, not by folder prefixes (e.g. alignment can happen before or after sorting).

## Adapting it
Everything is resolved through two small classes, so a different convention is a config change, not a code change:

```python
from np_ephys.layout import SessionPaths, HistologyPaths, ensure_dir

sp = SessionPaths.from_root("NP09_R1", "/data/MyExperiment")            # <root>/NP09/NP09_R1
sp = SessionPaths.from_root("NP09_R1", "/data", template="{session}")   # flat: <root>/NP09_R1
sp = SessionPaths.from_root("NP09_R1", "/data", names={"raw": "cbin"})  # rename a folder
hp = HistologyPaths.from_root("NP09", "/data")                          # <root>/NP09/histology/{downsampled,...}

sp.raw, sp.processed, sp.alignment, sp.derived, sp.sync_data, sp.unit_tables, sp.quality_metrics
sp.kilosort_dir(0), sp.analyzer_zarr(0, "clean"), sp.ibl_export(1)
ensure_dir(sp.sync_data)                                                # only when you write there
```

The same settings live in the TOML config (`[paths] root`, `session_template`, `histology_template`, `[paths.names]`;
see `config/example_config.toml`); `SessionPaths.from_config(cfg, session)` and `HistologyPaths.from_config(cfg, animal)`
read them. Environment variable `NP_EPHYS_ROOT` overrides `[paths].root`.

## Optional CLI
`np-setup-folders --root /data --session NP09_R1` prints where each folder is and whether it exists. Folders are created only
when you name them and pass `--create`, e.g. `np-setup-folders --root /data --session NP09_R1 --make raw tracks --create`.
