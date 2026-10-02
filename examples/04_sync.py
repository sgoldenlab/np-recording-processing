"""04 - detect sync pings on the sync channel and derive the experiment window (and segment offsets).

Real use:   python examples/04_sync.py --config my.toml --session NP01_R1 --probe 0
Synthetic:  python examples/04_sync.py --demo     (fake 2-segment recording with 1 Hz pings; no SpikeInterface needed)
"""
import numpy as np
from _common import config, parse_args

from np_ephys import sync
from np_ephys.config import recordings_from_config
from np_ephys.io_raw import find_raw_files, get_segment_boundaries
from np_ephys.layout import SessionPaths

FS = 1000.0


class FakeSyncRecording:
    """Minimal stand-in for a SpikeInterface recording (last channel = sync, 0/100 square wave)."""

    def __init__(self, ping_samples, n):
        x = np.zeros(n, dtype="int16")
        for p in ping_samples:
            x[p:p + 500] = 100
        self.traces = np.stack([np.zeros(n, dtype="int16"), x], axis=1)

    def get_num_samples(self):
        return self.traces.shape[0]

    def get_sampling_frequency(self):
        return FS

    def get_traces(self, start_frame=None, end_frame=None, return_scaled=False):
        return self.traces[start_frame:end_frame]


def demo():
    lens, local = [60_000, 40_000], []
    for k, n in enumerate(lens):
        pings = np.arange(1000, n - 1500, 1000) + k  # 1 Hz
        samples, times = sync.get_sync_edges(FakeSyncRecording(pings, n), threshold=40, chunk_duration_s=7.3)
        # an edge is reported at the last LOW sample before the rise (index i: x[i] < thr <= x[i+1])
        assert np.array_equal(samples, pings - 1), "edge detection lost/duplicated pings at chunk boundaries"
        local.append(samples)
        print(f"segment {k}: {samples.size} pings, window {sync.experiment_window(samples)}")
    bounds = sync.segment_boundaries_from_lengths(lens, FS)
    print("global windows:", sync.segment_sync_windows(bounds, local))
    print("periods:", sync.extract_sync_periods(local[0] / FS))


def main():
    args = parse_args(__doc__, lambda p: p.add_argument("--probe", type=int, default=0))
    if args.demo:
        return demo()
    cfg = config(args)
    s = cfg.get("sync", {})
    name, props = next(iter(recordings_from_config(cfg, args.session).items()))
    paths = SessionPaths.from_config(cfg, name)
    files = find_raw_files(paths.raw, name, concatenate=props["concatenate"])[args.probe]
    if props["concatenate"]:
        bounds = get_segment_boundaries(files, cache_file=paths.sync_data / f"segment_boundaries_probe{args.probe}.json")
        local = []
        for k, f in enumerate(files):
            samples, _ = sync.get_recording_sync([f], paths.sync_data, args.probe, segment=k,
                                                 threshold=s.get("threshold"), overwrite=args.overwrite)
            local.append(samples if samples is not None else np.array([]))
        print(sync.segment_sync_windows(bounds, local, s.get("end_index", -2)))
    else:
        samples, times = sync.get_recording_sync(files, paths.sync_data, args.probe, threshold=s.get("threshold"),
                                                 overwrite=args.overwrite)
        print("pings:", None if samples is None else samples.size, "window:",
              None if samples is None else sync.experiment_window(samples, s.get("end_index", -2)))


if __name__ == "__main__":
    main()
