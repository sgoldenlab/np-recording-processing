import numpy as np
import polars as pl
import pytest

from np_ephys import events as ev

NEW_FMT = """time_ms,time_s,event,gas_concentration
0,0.0,Experiment Started,N/A
499,0.499,Gas measurement,0.1
1497,1.497,Gas measurement,0.2
1800,1.8,EVENT_1_ISO_on,0.2
2495,2.495,Gas measurement,0.3
2900,2.9,TONESET_0_tone_0,0.3
3493,3.493,Gas measurement,N/A
4491,4.491,Gas measurement,0.5
"""
NO_LABEL_FMT = """time_ms,time_s,event,gas_concentration,video_frame
0,0.0,Experiment Started,N/A,0
499,0.499,Gas measurement,0.1,15
1000,1.0,Event_00,0.1,30
"""


def test_load_and_parse(tmp_path):
    f = tmp_path / "e.csv"
    f.write_text(NEW_FMT)
    df = ev.parse_event_labels(ev.load_events(f))
    assert df.height == 8
    assert df["time_s"].to_list()[1] == pytest.approx(0.499)
    row = df.filter(pl.col("event") == "EVENT_1_ISO_on").row(0, named=True)
    assert (row["event_type"], row["event_num"], row["event_label"]) == ("event", 1, "ISO on")
    row = df.filter(pl.col("event") == "TONESET_0_tone_0").row(0, named=True)
    assert (row["event_type"], row["event_num"], row["event_label"]) == ("toneset", 0, "tone_0")
    assert df.filter(pl.col("event_type") == "gas").height == 5
    assert df.filter(pl.col("event") == "Gas measurement")["gas_concentration"].null_count() == 1
    assert df["event_label"][0] == "Experiment Started"


def test_old_format_with_video_frame(tmp_path):
    f = tmp_path / "e.csv"
    f.write_text(NO_LABEL_FMT)
    df = ev.parse_event_labels(ev.load_events(f))
    assert "video_frame" in df.columns
    r = df.filter(pl.col("event") == "Event_00").row(0, named=True)
    assert r["event_type"] == "event" and r["event_num"] == 0 and r["event_label"] == "unlabeled"


def test_three_column_format(tmp_path):
    f = tmp_path / "e.csv"
    f.write_text("Time_s,Event,Sensor_Value\n0.5,Gas measurement,0.1\n1.5s,EVENT_2_fp,0.2\n")
    df = ev.load_events(f)
    assert df["time_ms"].to_list() == [500.0, 1500.0]


def test_frames_and_alignment(tmp_path):
    ef, ff = tmp_path / "e.csv", tmp_path / "f.csv"
    ef.write_text(NEW_FMT)
    t = np.arange(0, 5, 1 / 30)
    pl.DataFrame({"Frame_ID": np.arange(t.size), "Hardware_Timestamp_ns": (1e12 + t * 1e9).astype("int64"),
                  "System_Arrival_Time_s": 1.7e9 + t}).write_csv(ff)
    events = ev.parse_event_labels(ev.load_events(ef))
    frames = ev.load_frames(ff)
    assert frames["frame_time_s"][0] == 0
    m = ev.create_sync_frame_mapping(events, frames)
    assert m.height == 5 and m["time_offset_ms"].abs().max() < 17
    al = ev.align_events(events, m)
    assert al.height == 3  # Experiment Started, EVENT, TONESET
    assert al.filter(pl.col("event_label") == "ISO on")["video_frame"][0] in range(30, 90)
    no_video = ev.align_events(events, None)
    assert no_video["video_frame"].null_count() == no_video.height


def test_add_recording_time(tmp_path):
    f = tmp_path / "e.csv"
    f.write_text(NEW_FMT)
    events = ev.parse_event_labels(ev.load_events(f))
    log = events.filter(pl.col("event_type") == "gas")["time_s"].to_numpy()
    rec = 100.0 + 1.00002 * log
    out = ev.add_recording_time(events, rec)
    iso = out.filter(pl.col("event_label") == "ISO on")["recording_time_s"][0]
    assert iso == pytest.approx(100.0 + 1.00002 * 1.8, abs=1e-6)
    # one detected ping too few -> warning, truncation
    with pytest.warns(UserWarning):
        ev.add_recording_time(events, rec[:-1])
