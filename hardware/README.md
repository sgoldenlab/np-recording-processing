# hardware/ - Arduino + Raspberry Pi sync rig

Generates the sync pings that let the behaviour/event log, the video and the Neuropixels recording share a clock.

```
Raspberry Pi  --USB serial 9600-->  Arduino  --pin 7 (1 Hz square wave)-->  Neuropixels DAQ sync input
     |  \--USB serial 115200--> AA-8000 gas analyzer                         (last channel of the .ap stream)
     |  \--USB camera (V4L2/OpenCV)
     \--- 3 push buttons: START (GPIO17), EVENT (GPIO27), STOP (GPIO22)   (BCM numbering, pull-down)
```

## Files
| File | Purpose |
|---|---|
| `arduino/arduino_sync_with_RPi_v2.ino` | Arduino sync sketch. Square wave on pin 7, LED on pin 13 (on between pings). |
| `raspberry_pi/sync_script_v6.py` | Pi logger (buttons, pings, gas analyzer, camera). |
| `camera/load_cam_settings.sh` | `v4l2-ctl` settings (manual exposure 200, fixed white balance 2800, 30 fps). |
| `camera/camera-settings_summary.txt` | `v4l2-ctl --all` dump of the camera used (UVC, MJPG 1280x720). |

**Not included:** the Pi script that writes `EVENT_<n>_<label>` / `TONESET_<n>_<label>` rows and `frames.csv`. `np_ephys.events` parses those formats, but the producing script must be supplied separately.

## Wiring
* Arduino pin 7 -> DAQ digital/sync input of the Neuropixels acquisition system; common ground. 3.3-5 V logic, so check your DAQ's input range.
* Arduino USB -> Pi (`/dev/ttyACM0`). AA-8000 USB-serial -> Pi (`/dev/ttyUSB0`). Camera -> Pi USB (`/dev/video0`).
* Buttons between 3.3 V and GPIO 17 / 27 / 22 (internal pull-downs are enabled).
* Device names are the Pi defaults and are hard-coded at the top of `sync_script_v6.py`; edit for other hardware.

## Serial protocol (Pi <-> Arduino, 9600 baud, 1 byte commands)
| Byte | Direction | Effect |
|---|---|---|
| `S` | Pi -> Arduino | Start: zero the Arduino clock (`millis()`), begin pinging. |
| `R` | Pi -> Arduino | Read: Arduino answers one line with ms since `S` (used to timestamp button events). |
| `E` | Pi -> Arduino | End: answers one line with ms since `S`, stops pinging. |
| `<ms>\n` | Arduino -> Pi | Sent at each ping rising edge (ms since `S`). |

Ping: the pin goes HIGH for 499 ms and LOW for 499 ms (about 1.002 s period, ~1 Hz). `millis()` timing means individual pings have ~1 ms jitter and the Arduino clock drifts slightly relative to the DAQ clock; this is why event times are mapped with piecewise-linear interpolation between ping pairs (`np_ephys.sync.events_to_recording_time`).
Caveat: the answer to `R`/`E` is read with `readline()`, so if a ping line arrives at the same moment the event can pick up the ping's timestamp instead; the error is below one period but check `events.csv` for duplicated times.

## Output files
**`<name>.csv` (the "events" log)**, one row appended per event:
`time_ms, time_s, event, gas_concentration, video_frame`
* `time_ms`: Arduino clock, ms since START (the raw serial string, may carry a trailing `\r\n`; `np_ephys.events.load_events` strips it).
* `event`: `Experiment Started`, `Gas measurement` (one per ping), `Event_NN` (button, v6; or `EVENT_<n>_<label>`, `TONESET_<n>_<label>`), `Experiment Stopped`.
* `gas_concentration`: the AA-8000 reply or `N/A`; `video_frame`: frame counter at logging time.

**Ping k in the recording = "Gas measurement" row k** (`np_ephys.sync.match_ping_counts` checks the counts and trims extra edges).

**`<name>_frames.csv`** (only with scripts that record video frames): `Frame_ID, Hardware_Timestamp_ns, System_Arrival_Time_s`, one row per video frame; `np_ephys.events.create_sync_frame_mapping` links ping rows to frames.

**Pings in the recording** (`ping_samples_*.npy`, `ping_times_*.npy`) are written by `np_ephys.sync.get_recording_sync` on the analysis machine, not by the Pi; names encode probe, segment, threshold and range.

## AA-8000 gas analyzer notes
* 115200 baud, query `READ:MEAS?\r` (a carriage return is required; `\r\n` also works). The reply is one ASCII line with the concentration.
* `read_gas` treats replies shorter than 4 characters or starting with `.` as invalid and logs `N/A`; the read has a 0.9 s timeout, so a missing analyzer slows every ping row.
* Reading the analyzer happens in the same loop as the pings; keep the loop delay (50 ms) small relative to the 499 ms half period.
* Units and calibration are set on the analyzer, not recorded in the log; note them in your session metadata.

## Using the camera script
`bash camera/load_cam_settings.sh` before recording (needs `v4l2-ctl`). The script asks for an experiment name and whether to save csv/video; it records 640x480 MJPG to `<name>.mp4`.
Dependencies on the Pi: `RPi.GPIO`, `pyserial`, `opencv-python`, `pandas` (not installed by this package).
