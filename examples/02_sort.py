"""02 - Kilosort4 through SpikeInterface (single shank or by shank), one folder per probe.

Real use:   python examples/02_sort.py --config my.toml --session NP01_R1      (needs a CUDA GPU + np-recording-processing[ephys,sorting])
Synthetic:  python examples/02_sort.py --demo     (no GPU: prints the parameters and the probe plan only)
"""
from _common import config, parse_args, scratch

from np_ephys.config import recordings_from_config
from np_ephys.layout import SessionPaths
from np_ephys.sorting import build_ks_params, prepare_probe_configuration, sort_session


def main():
    args = parse_args(__doc__)
    if args.demo:
        params = build_ks_params()
        print("Kilosort4 parameters (defaults):", params)
        print("by-shank plan for 2 probes, multiple_shanks=[False, True]:", prepare_probe_configuration([False, True], 2))
        print("scratch folder would be:", scratch("sort"))
        return
    cfg = config(args)
    params = build_ks_params(cfg.get("kilosort"))
    for name, props in recordings_from_config(cfg, args.session).items():
        paths = SessionPaths.from_config(cfg, name)
        res = sort_session(paths, props, params, overwrite=args.overwrite)
        print(name, res["status"], res["errors"])


if __name__ == "__main__":
    main()
