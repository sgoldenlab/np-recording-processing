"""05 - spike table (unit_id, sample_index) -> windowed per-unit table with firing rates.

Real use:   python examples/05_spiketimes.py --config my.toml --session NP01_R1 --probe 0   (needs steps 02-04)
Synthetic:  python examples/05_spiketimes.py --demo
"""
import numpy as np
import polars as pl
from _common import config, parse_args

from np_ephys import spiketimes as st
from np_ephys.config import recordings_from_config
from np_ephys.layout import SessionPaths, ensure_dir

FS = 30000.0


def main():
    args = parse_args(__doc__, lambda p: p.add_argument("--probe", type=int, default=0))
    if args.demo:
        rng = np.random.default_rng(0)
        unit_ids = [3, 7, 12]  # NOT 0..N-1: curated sortings keep their ids
        idx = rng.integers(0, 3, 3000)
        samples = np.sort(rng.integers(0, int(100 * FS), 3000))
        spikes = st.spikes_to_frame(samples, idx, unit_ids)
        windows = [dict(start_sync=int(10 * FS), end_sync=int(40 * FS)), dict(start_sync=int(60 * FS), end_sync=int(90 * FS))]
        meta = pl.DataFrame({"unit_id": unit_ids, "cell_type": ["putative_pyramidal", "putative_interneuron", "unclassified"]})
        print(st.build_unit_table(spikes, meta, FS, windows=windows).drop("spike_times", "spike_times_sec"))
        return
    import spikeinterface.full as si

    from np_ephys import sync
    cfg = config(args)
    name, _ = next(iter(recordings_from_config(cfg, args.session).items()))
    paths = SessionPaths.from_config(cfg, name)
    sorting = si.read_kilosort(paths.kilosort_dir(args.probe) / "sorter_output")
    spikes = st.spikes_from_sorting(sorting)
    th = cfg.get("sync", {}).get("threshold")
    samples = None
    from np_ephys.sync import sync_cache_paths
    sp, _ = sync_cache_paths(paths.sync_data, args.probe, threshold=th)
    if sp.exists():
        samples = np.load(sp)
    if samples is None:
        raise SystemExit(f"run 04_sync.py first (expected {sp})")
    win = sync.experiment_window(samples, cfg.get("sync", {}).get("end_index", -2))
    table = st.build_unit_table(spikes, spikes.select("unit_id").unique(), sorting.get_sampling_frequency(), windows=[win])
    ensure_dir(paths.unit_tables)
    table.drop("spike_times", "spike_times_sec").write_csv(paths.unit_tables / f"{name}_probe{args.probe}_units.csv")


if __name__ == "__main__":
    main()
