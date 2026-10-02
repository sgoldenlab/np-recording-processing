import pytest

from np_ephys.io_raw import find_raw_files


def _touch(p):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"")


def test_concatenated_sorted_and_grouped(tmp_path):
    for seg in ("NP09_R1_trials_g0", "NP09_R1_g0", "NP09_R1_g1"):
        for probe in (1, 0):
            _touch(tmp_path / seg / f"{seg}_t0.imec{probe}.ap.cbin")
    got = find_raw_files(tmp_path, "NP09_R1", concatenate=True)
    assert list(got) == [0, 1]
    names = [p.parent.name for p in got[0]]
    assert names == ["NP09_R1_g0", "NP09_R1_g1", "NP09_R1_trials_g0"]
    got = find_raw_files(tmp_path, "NP09_R1", concatenate=True, order=["trials", "_g1", "_g0"])
    assert [p.parent.name for p in got[1]] == ["NP09_R1_trials_g0", "NP09_R1_g1", "NP09_R1_g0"]


def test_single_flat_and_error(tmp_path):
    _touch(tmp_path / "A_R1_g0_t0.imec0.ap.cbin")
    assert list(find_raw_files(tmp_path, "A_R1")) == [0]
    _touch(tmp_path / "A_R1_g1_t0.imec0.ap.cbin")
    with pytest.raises(ValueError, match="concatenate"):
        find_raw_files(tmp_path, "A_R1")


def test_natural_sort(tmp_path):
    for t in (10, 2, 1):
        _touch(tmp_path / f"A_R1_g0_t{t}.imec0.ap.cbin")
    got = find_raw_files(tmp_path, "A_R1", concatenate=True)[0]
    assert [p.name.split("_")[3].split(".")[0] for p in got] == ["t1", "t2", "t10"]
