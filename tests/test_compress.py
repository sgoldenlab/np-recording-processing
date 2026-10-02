import numpy as np
import pytest

mtscomp = pytest.importorskip("mtscomp")
from np_ephys.compress import compress_session  # noqa: E402


def _fake_run(root, name, n_ch=5, n_samp=6000, seed=0):
    d = root / f"{name}_g0" / f"{name}_g0_imec0"
    d.mkdir(parents=True)
    rng = np.random.default_rng(seed)
    x = (rng.standard_normal((n_samp, n_ch)) * 20).astype("int16")
    x.tofile(d / f"{name}_g0_t0.imec0.ap.bin")
    (d / f"{name}_g0_t0.imec0.ap.meta").write_text(
        f"imSampRate=30000\nnSavedChans={n_ch}\nsnsApLfSy={n_ch-1},0,1\n")
    return root / f"{name}_g0", x


def test_compress_roundtrip(tmp_path):
    seg, x = _fake_run(tmp_path / "in", "T_R1")
    outs = compress_session([seg], tmp_path / "out", concatenate=False,
                            job_kwargs=dict(n_threads=1, chunk_duration=0.1))
    assert outs[0].name == "T_R1_g0_t0.imec0.ap.cbin" and (tmp_path / "out" / "T_R1_g0_t0.imec0.ap.meta").exists()
    r = mtscomp.decompress(outs[0], outs[0].with_suffix(".ch"))
    assert np.array_equal(r[:], x)


def test_concatenated_gets_subfolders_and_is_findable(tmp_path):
    s0, _ = _fake_run(tmp_path / "in", "T_R1", seed=1)
    s1, _ = _fake_run(tmp_path / "in", "T_R1_trials", seed=2)
    out = compress_session([s0, s1], tmp_path / "out", concatenate=True,
                           job_kwargs=dict(n_threads=1, chunk_duration=0.1))
    assert {p.parent.name for p in out} == {"T_R1_g0", "T_R1_trials_g0"}
    from np_ephys.io_raw import find_raw_files
    got = find_raw_files(tmp_path / "out", "T_R1", concatenate=True)
    assert [p.parent.name for p in got[0]] == ["T_R1_g0", "T_R1_trials_g0"]
