"""01 - compress SpikeGLX AP files (*.ap.bin) to mtscomp .cbin/.ch + copy .meta.

Real use:   python examples/01_compress.py --source /path/to/SpikeGLX/NP01_R1_g0 [more segment folders] --session NP01_R1
Synthetic:  python examples/01_compress.py --demo
Needs: pip install "np-recording-processing[compress]"
"""
import numpy as np
from _common import config, parse_args, scratch

from np_ephys.compress import compress_session, default_job_kwargs
from np_ephys.config import recordings_from_config
from np_ephys.layout import SessionPaths


def fake_spikeglx_run(root, name, n_ch=5, n_samp=6000, seed=0):
    d = root / f"{name}_g0" / f"{name}_g0_imec0"
    d.mkdir(parents=True)
    x = (np.random.default_rng(seed).standard_normal((n_samp, n_ch)) * 20).astype("int16")
    x.tofile(d / f"{name}_g0_t0.imec0.ap.bin")
    (d / f"{name}_g0_t0.imec0.ap.meta").write_text(f"imSampRate=30000\nnSavedChans={n_ch}\nsnsApLfSy={n_ch-1},0,1\n")
    return root / f"{name}_g0"


def main():
    args = parse_args(__doc__, lambda p: p.add_argument("--source", nargs="*", default=[],
                                                        help="SpikeGLX run folders ({name}_g0) of the session"))
    if args.demo:
        tmp = scratch("compress")
        seg = fake_spikeglx_run(tmp / "spikeglx", "DEMO_R1")
        out = compress_session([seg], tmp / "raw", job_kwargs=dict(n_threads=1, chunk_duration=0.1))
        print("compressed:", [str(p) for p in out])
        return
    cfg = config(args)
    name, props = next(iter(recordings_from_config(cfg, args.session).items()))
    paths = SessionPaths.from_config(cfg, name)
    kw = default_job_kwargs(cfg.get("job", {}).get("fraction_of_cores", 0.8), cfg.get("job", {}).get("chunk_duration", 5))
    out = compress_session(args.source, paths.raw, concatenate=props["concatenate"], job_kwargs=kw,
                           overwrite=args.overwrite)
    print("compressed:", *out, sep="\n  ")


if __name__ == "__main__":
    main()
