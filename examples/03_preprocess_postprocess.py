"""03 - SortingAnalyzer, extensions, quality metrics, curation, cell-type split.

Real use:   python examples/03_preprocess_postprocess.py --config my.toml --session NP01_R1 --probe 0
            (loads the raw .cbin, preprocesses it, reads the Kilosort4 output, writes analyzers/metrics to <session>/processed)
Synthetic:  python examples/03_preprocess_postprocess.py --demo   (SpikeInterface ground-truth recording, 16 channels, 30 s)
Needs: np-recording-processing[ephys]
"""
from _common import config, parse_args

from np_ephys import analyzer as an
from np_ephys import curation as cu
from np_ephys.config import recordings_from_config
from np_ephys.layout import SessionPaths, ensure_dir


def postprocess(sorting, recording, folder, min_spikes, query, ptv_ms, name, probe):
    sorting = an.clean_sorting(sorting, recording, min_spikes=min_spikes)
    analyzer = an.create_analyzer(sorting, recording, folder=folder)
    an.compute_extensions(analyzer, an.EXTENSION_ORDER, verbose=False)
    metrics = an.compute_quality_metrics(analyzer, an.QUALITY_METRICS_FULL, recording_name=name, probe_id=probe)
    good = cu.query_curate(metrics, query)
    print(f"{len(good)}/{len(metrics)} units pass: {query}")
    tm = analyzer.get_extension("template_metrics").get_data()
    cells = cu.classify_cell_type(tm, ptv_threshold_ms=ptv_ms)
    print(cells["cell_type"].value_counts().to_dict())
    return analyzer, metrics, cells


def main():
    args = parse_args(__doc__, lambda p: p.add_argument("--probe", type=int, default=0))
    if args.demo:
        from spikeinterface.core import generate_ground_truth_recording

        rec, sorting = generate_ground_truth_recording(durations=[30.0], num_channels=16, num_units=6, seed=0)
        postprocess(sorting, rec, None, 20, "snr > 0", 0.4, "DEMO_R1", 0)
        return
    import spikeinterface.full as si

    from np_ephys.io_raw import find_raw_files, load_recording
    from np_ephys.preprocess import kilosort_like_preprocess, save_recording_zarr

    cfg = config(args)
    name, props = next(iter(recordings_from_config(cfg, args.session).items()))
    paths = SessionPaths.from_config(cfg, name)
    files = find_raw_files(paths.raw, name, concatenate=props["concatenate"])[args.probe]
    rec = load_recording(files, include_sync=False, concatenate=props["concatenate"])
    pre = kilosort_like_preprocess(rec)
    ensure_dir(paths.processed)
    save_recording_zarr(pre, paths.preprocessed_zarr(args.probe, props["concatenate"]))
    sorting = si.read_kilosort(paths.kilosort_dir(args.probe) / "sorter_output")
    cur = cfg.get("curation", {})
    analyzer, metrics, cells = postprocess(
        sorting, pre, paths.analyzer_zarr(args.probe), cfg.get("analyzer", {}).get("min_spikes", 500),
        cur.get("query", cu.DEFAULT_QUERY), cur.get("ptv_threshold_ms", 0.4), name, args.probe)
    ensure_dir(paths.quality_metrics)
    metrics.to_csv(paths.quality_metrics / f"{name}_probe{args.probe}_metrics.csv")
    an.export_report(analyzer, paths.processed / f"report_probe{args.probe}")


if __name__ == "__main__":
    main()
