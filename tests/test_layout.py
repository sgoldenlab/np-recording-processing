from pathlib import Path

import pytest

from np_ephys.config import load_config, recordings_from_config
from np_ephys.layout import (HistologyPaths, SessionPaths, animal_of, ensure_dir, ensure_parent, main,
                             probe_from_name)


def test_names():
    assert animal_of("NP09_R1") == "NP09"
    assert probe_from_name("NP09_R1_g0_t0.imec1.ap.cbin") == 1
    assert probe_from_name("nothing") is None


def test_resolving_paths_creates_nothing(tmp_path):
    sp = SessionPaths.from_root("NP09_R1", tmp_path)
    hp = HistologyPaths.from_root("NP09", tmp_path)
    _ = (sp.raw, sp.processed, sp.alignment, sp.derived, sp.sync_data, sp.unit_tables, sp.quality_metrics,
         sp.kilosort_dir(0), sp.ibl_export(1), hp.downsampled, hp.registered, hp.stacks, hp.tracks)
    assert list(tmp_path.iterdir()) == []
    assert sp.existing() == []


def test_default_layout(tmp_path):
    sp = SessionPaths.from_root("NP09_R1", tmp_path)
    assert sp.root == tmp_path / "NP09" / "NP09_R1"
    assert sp.raw == sp.root / "raw"
    assert sp.sync_data == sp.root / "derived" / "sync"
    assert sp.unit_tables == sp.root / "derived" / "units"
    assert sp.quality_metrics == sp.root / "processed" / "metrics"
    hp = HistologyPaths.from_root("NP09", tmp_path)
    assert hp.root == tmp_path / "NP09" / "histology"
    assert (hp.downsampled.name, hp.registered.name, hp.stacks.name, hp.tracks.name) == (
        "downsampled", "registered", "stacks", "tracks")


def test_lazy_creation(tmp_path):
    sp = SessionPaths.from_root("NP09_R1", tmp_path)
    ensure_dir(sp.sync_data)
    assert sp.existing() == [sp.derived]
    assert not sp.raw.exists()
    f = ensure_parent(sp.unit_tables / "x.csv")
    assert f.parent.is_dir() and not f.exists()


def test_custom_template_and_names(tmp_path):
    sp = SessionPaths.from_root("NP09_R1", tmp_path, template="{session}", names={"raw": "cbin", "sync": "pings"})
    assert sp.root == tmp_path / "NP09_R1"
    assert sp.raw.name == "cbin"
    assert sp.sync_data == sp.derived / "pings"
    hp = HistologyPaths.from_root("NP09", tmp_path, template="histology/{animal}", names={"tracks": "lasagna"})
    assert hp.tracks == tmp_path / "histology" / "NP09" / "lasagna"


def test_cli_prints_only_by_default(tmp_path, capsys):
    main(["--root", str(tmp_path), "--session", "NP01_R1"])
    assert list(tmp_path.iterdir()) == []
    assert "raw" in capsys.readouterr().out
    main(["--root", str(tmp_path), "--session", "NP01_R1", "--make", "raw", "tracks"])
    assert list(tmp_path.iterdir()) == []                          # still dry without --create
    main(["--root", str(tmp_path), "--session", "NP01_R1", "--make", "raw", "tracks", "--create"])
    assert (tmp_path / "NP01" / "NP01_R1" / "raw").is_dir()
    assert (tmp_path / "NP01" / "histology" / "tracks").is_dir()
    assert not (tmp_path / "NP01" / "NP01_R1" / "processed").exists()
    with pytest.raises(SystemExit):
        main(["--root", str(tmp_path), "--session", "A_R1", "--make", "bogus"])


def test_session_paths_and_config(tmp_path):
    cfg_file = tmp_path / "c.toml"
    cfg_file.write_text(
        '[paths]\nroot = "/data"\n[paths.names]\nraw = "compressed"\n'
        '[experiment.recordings]\nNP09_R1 = { concatenate = true }\nNP09_R2 = {}\n'
    )
    cfg = load_config(cfg_file)
    recs = recordings_from_config(cfg)
    assert recs["NP09_R1"] == {"concatenate": True, "multiple_shanks": False}
    assert list(recordings_from_config(cfg, only="NP09_R2")) == ["NP09_R2"]
    sp = SessionPaths.from_config(cfg, "NP09_R1")
    assert sp.root == Path("/data/NP09/NP09_R1")
    assert sp.raw.name == "compressed" and sp.processed.name == "processed"
    assert HistologyPaths.from_config(cfg, "NP09").tracks == Path("/data/NP09/histology/tracks")
    assert sp.kilosort_dir(1).name == "kilosort4_probe1"
    assert sp.preprocessed_zarr(0, True).name == "preprocessed_recording_probe0_concatenated.zarr"
    assert sp.analyzer_zarr(0, "clean").name == "sorting_analyzer_clean_probe0.zarr"
    assert sp.analyzer_zarr(2).name == "sorting_analyzer_probe2.zarr"


def test_env_root_override(tmp_path, monkeypatch):
    f = tmp_path / "c.toml"
    f.write_text('[paths]\nroot = "/a"\n')
    monkeypatch.setenv("NP_EPHYS_ROOT", "/b")
    assert load_config(f)["paths"]["root"] == "/b"
