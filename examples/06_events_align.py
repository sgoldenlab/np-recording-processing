"""06 - align the Pi event log to the video frames and to the recording clock.

Real use:   python examples/06_events_align.py --events X_events.csv [--frames X_frames.csv] --pings ping_times_probe0_thrauto.npy
Synthetic:  python examples/06_events_align.py --demo
"""
import numpy as np
import polars as pl
from _common import parse_args, scratch

from np_ephys import events as ev


def synth(tmp):
    rows = ["time_ms,time_s,event,gas_concentration", "0,0.0,Experiment Started,N/A"]
    for k in range(1, 61):
        rows.append(f"{499 + (k - 1) * 1002},{(499 + (k - 1) * 1002) / 1000},Gas measurement,0.{k % 10}")
    rows.insert(20, "17000,17.0,EVENT_1_ISO_on,0.1")
    rows.insert(40, "37000,37.0,TONESET_0_tone_0,0.1")
    (tmp / "demo_events.csv").write_text("\n".join(rows) + "\n")
    t = np.arange(0, 62, 1 / 30)
    pl.DataFrame({"Frame_ID": np.arange(t.size), "Hardware_Timestamp_ns": (1e12 + t * 1e9).astype("int64"),
                  "System_Arrival_Time_s": 1.7e9 + t}).write_csv(tmp / "demo_frames.csv")
    log = np.arange(60) * 1.002 + 0.499
    np.save(tmp / "demo_pings.npy", 120.0 + 1.00003 * log)  # recording clock: offset 120 s, 30 ppm drift
    return tmp / "demo_events.csv", tmp / "demo_frames.csv", tmp / "demo_pings.npy"


def main():
    def extra(p):
        p.add_argument("--events"); p.add_argument("--frames"); p.add_argument("--pings")  # noqa: E702
    args = parse_args(__doc__, extra)
    if args.demo:
        e, f, p = synth(scratch("events"))
    else:
        if not (args.events and args.pings):
            raise SystemExit("--events and --pings are required (or use --demo)")
        e, f, p = args.events, args.frames, args.pings
    events = ev.parse_event_labels(ev.load_events(e))
    frames = ev.load_frames(f) if f else None
    sync_map = ev.create_sync_frame_mapping(events, frames) if frames is not None else None
    aligned = ev.align_events(events, sync_map)
    rec_time = ev.add_recording_time(events, np.load(p))
    out = aligned.join(rec_time.select("event", pl.col("time_s").alias("sync_time_s"), "recording_time_s"),
                       on=["event", "sync_time_s"], how="left")
    print(out.select("event_label", "event_type", "sync_time_s", "video_frame", "recording_time_s"))


if __name__ == "__main__":
    main()
