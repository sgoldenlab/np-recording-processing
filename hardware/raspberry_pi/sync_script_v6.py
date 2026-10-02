"""Raspberry Pi experiment logger (sync_script_v6): buttons, Arduino sync pings, AA-8000 gas analyzer, camera.

Runs on the Pi during a recording. Press START (GPIO 17) -> sends 'S' to the Arduino, which starts the
1 Hz sync square wave to the Neuropixels DAQ and prints each ping time (ms since 'S') on serial; every
ping line is logged here as a "Gas measurement" row together with an AA-8000 reading and the current
video frame number. EVENT button (GPIO 27) logs "Event_NN"; STOP (GPIO 22) sends 'E' and ends the run.
Output: <name>.csv (columns time_ms,time_s,event,gas_concentration,video_frame) and <name>.mp4.
See hardware/README.md. Serial ports (/dev/ttyACM0, /dev/ttyUSB0) and the camera index are the Pi's
defaults; edit the constants below for another setup.
"""
import RPi.GPIO as GPIO
import time
import pandas as pd
import serial
import glob
import cv2  # OpenCV for ArduCam
from threading import Thread
from pathlib import Path

# GPIO Pin Assignments
START_BUTTON_PIN = 17  # Start experiment
TOE_PINCH_BUTTON_PIN = 27  # Log "Toe Pinch" event
STOP_BUTTON_PIN = 22  # Stop experiment

# Setup GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setup(START_BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
GPIO.setup(TOE_PINCH_BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
GPIO.setup(STOP_BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)

# Initialize Serial Connections
arduino = serial.Serial('/dev/ttyACM0', 9600, timeout=1)
gas_analyzer = serial.Serial('/dev/ttyUSB0', 115200, timeout=.9)
time.sleep(2)  # Allow devices to initialize

# Function to get the next available filename


def get_new_filename(prefix, extension):
    existing_files = glob.glob(f"{prefix}_*.{extension}")
    valid_numbers = []

    for f in existing_files:
        try:
            # Extract the number part
            number = int(f.split('_')[1].split('.')[0])
            valid_numbers.append(number)
        except (ValueError, IndexError):
            continue  # Ignore files that don't match the expected pattern

    next_number = max(valid_numbers) + 1 if valid_numbers else 1
    return f"{prefix}_{next_number}.{extension}"


# Generate filenames for log and video
default_name = False
if default_name:
    csv_filename = get_new_filename("experiment", "csv")
    video_filename = str(Path(csv_filename).with_suffix(
        '.avi'))  # to make sure it shares expt name
else:
    name = input("please enter name of experiment (without extension): ")
    csv_filename = f'{name}.csv'
    # video_filename = f'{name}.avi'
    video_filename = f'{name}.mp4'


# Initialize CSV file
column_names = ["time_ms", "time_s", "event",
                "gas_concentration", "video_frame"]
data = pd.DataFrame(columns=column_names)
# pd.DataFrame(columns=column_names).to_csv(csv_filename, index=False)

match input('save csv (y/n)? default=yes') or 'y':
    case 'n' | 'N' | 'no':
        save_csv = False
    case _:
        save_csv = True
        data.to_csv(csv_filename, index=False)


def write_gas(port, msg):
    """
    Sends a message to the specified port.

    Args:
        port (serial.Serial): The serial port to which the message will be sent.
        msg (str): The message to be sent.

    Returns:
        None
    """
    port.write(msg.encode())


def read_ser(port):
    """
    Reads a line from the specified serial port.

    Args:
        port (serial.Serial): The serial port object to read from.

    Returns:
        bytes: The line read from the serial port.
    """
    line = port.readline()

    return line


def read_gas(port, meas):
    """
    Reads a gas measurement from the specified serial port.
    This function sends a command to the AA-8000 analyzer to read the gas measurement.
    It ensures the command includes a return carriage character, flushes the port, 
    and reads the response. If the response is invalid (less than 4 characters or 
    starts with a period), it returns the provided default measurement.
    Args:
        port (serial.Serial): The serial port connected to the gas analyzer.
        meas (str): The default measurement to return if the reading is invalid.
    Returns:
        str: The gas measurement read from the analyzer, or the default measurement if invalid.
    """

    # AA-8000 analyzer requires return carriage (and maybe also newline) character
    write_gas(port, "READ:MEAS?\r")
    port.flush()
    reading = read_ser(port).decode().strip()
    if (len(reading) < 4) or (reading.startswith('.')):
        reading = meas
    return reading


def get_time_s(time_in, as_string: bool = False):
    time_s = float(time_in)/1000.
    if as_string:
        return f"{time_s:.2f}s"
    else:
        return time_s

# Function to log events


def log_event(experiment_time, event, concentration="N/A", video_frame=None, print_event=False):
    global save_csv
    experiment_time_s = get_time_s(experiment_time)
    if save_csv:
        df = pd.DataFrame([[experiment_time, experiment_time_s, event, concentration, video_frame]],
                          columns=column_names)
        df.to_csv(csv_filename, mode='a', header=False, index=False)

    if print_event:
        frame_info = f' ({float(video_frame)/30.0:.2f}s)' if video_frame else ''
        print(
            f"[{experiment_time_s:.2f}s] {event}\n\tGas Concentration: {concentration}\n\tVideo Frame: {video_frame}{frame_info}",
            "\n"
        )
    else:
        frame_info = f' ({float(experiment_time_s):.2f}s)' if video_frame else ''
        print(
            f"[{experiment_time_s:.2f}s] - Video Frame: {video_frame}{frame_info}\n")


# ArduCam Setup
# w, h = (1280, 720)
w, h = (640, 480)
camera = cv2.VideoCapture(0,
                          apiPreference=cv2.CAP_V4L2,
                          #   params=[
                          #       #   cv2.CAP_PROP_FRAME_WIDTH, w,
                          #       #   cv2.CAP_PROP_FRAME_HEIGHT, h,
                          #     #   cv2.CAP_PROP_FPS, 30,
                          #       #   cv2.CAP_PROP_FOURCC, 'MJPG',
                          #   ]
                          )

# Open ArduCam
camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Video format
camera.set(cv2.CAP_PROP_FPS, 30)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, w)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

time.sleep(1)
if not camera.isOpened():
    print('could not open camera')
    exit()
print('size', camera.get(cv2.CAP_PROP_FRAME_WIDTH),
      camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
print('fps', camera.get(cv2.CAP_PROP_FPS))
video_out = None  # Placeholder for video writer
global frame_number  # Initialize frame counter
frame_number = 0


def set_resolution(frame, width, height):
    return cv2.resize(frame, (width, height))


def record_video(w, h):
    """Continuously records video until stopped."""
    global video_out, frame_number, camera
    # w,h=(1280,720)
    while experiment_started:
        ret, frame = camera.read()
        if ret:
            # resized_frame = set_resolution(frame, w, h)
            # video_out.write(resized_frame)
            video_out.write(frame)
            frame_number += 1
        # time.sleep(0.03)  # 30 FPS limit


def display_feed_with_resolution(w=1280, h=720):
    """Display camera feed with resolution restriction."""
    global stop_cam
    cv2.namedWindow('Camera Feed', cv2.WINDOW_FULLSCREEN)
    cv2.setWindowProperty(
        'Camera Feed', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    while experiment_started:
        ret, frame = camera.read()
        if ret:
            cv2.imshow('Camera Feed', frame)
#        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit feed
#            stop_cam=True


save_video = True
match input("save video (y/n)? default=yes"):
    case 'n' | 'N' | 'no':
        save_video = False
    case _:
        pass

experiment_started = False
sync_started = False
start_time_pi = None
video_thread = None
event_num = 0
last_event = ''
print(f'saving video resolution:  {w} x {h}')
try:
    print("Waiting for start button press to begin experiment...")
    while True:
        current_time = time.time()

        # Start experiment
        if (GPIO.input(START_BUTTON_PIN) == GPIO.HIGH and not experiment_started):
            print("Starting Experiment...")
            arduino.write(b'S')  # Signal Arduino to start
            start_time_pi = time.time()  # Set experiment start time
            experiment_started = True
            log_event(0, "Experiment Started")

            if save_video:
                # Start video recording
                video_out = cv2.VideoWriter(
                    video_filename, fourcc, 30.0, (w, h))
                video_thread = Thread(target=record_video, args=(w, h))
                video_thread.start()
            else:
                # Display camera feed
                feed_thread = Thread(
                    target=display_feed_with_resolution, args=(640, 480))
                feed_thread.start()
            time.sleep(0.2)  # debounce

        # Log event
        if GPIO.input(TOE_PINCH_BUTTON_PIN) == GPIO.HIGH and experiment_started:
            experiment_time_pi = time.time() - start_time_pi
            arduino.write(b'R')
            arduino_data = arduino.readline().decode()  # .strip()
            gas_concentration = read_gas(gas_analyzer, "N/A")
            gas_analyzer.reset_output_buffer()
            log_event(arduino_data, f"Event_{event_num:02d}",
                      concentration=gas_concentration, video_frame=frame_number)
            event_num += 1
            last_event = f"[Event_{event_num:02d} at {get_time_s(arduino_data, as_string=True)}]"
            time.sleep(0.2)  # debounce
            print("\nLast Event:\t", last_event)

        # Stop experiment
        if GPIO.input(STOP_BUTTON_PIN) == GPIO.HIGH and experiment_started:
            arduino.write(b'E')  # Signal Arduino to start
            arduino_data = arduino.readline().decode()  # .strip()
            log_event(arduino_data,
                      "Experiment Stopped", video_frame=frame_number)
            print("Experiment stopped.")
            experiment_started = False
            break

        # sensor reading based on Arduino pings

        if experiment_started and arduino.in_waiting > 0:
            # print(arduino.in_waiting)
            arduino_data = arduino.readline().decode()  # .strip()
            try:
                # print(arduino_data)
                # arduino_time, _ = arduino_data.split(',')  # Ignore Arduino gas value
                # experiment_time_pi = current_time - start_time_pi

                gas_concentration = read_gas(gas_analyzer, "N/A")
                gas_analyzer.reset_output_buffer()

                log_event(arduino_data,
                          "Gas measurement", gas_concentration, frame_number)
                print("Last Event:\t", last_event, f'\n{"---"*10}')
                # print('frame: ', frame_number)
                # print(gas_concentration)

            except ValueError:
                print("Invalid data from Arduino:", arduino_data)
        time.sleep(0.05)  # Small delay

except KeyboardInterrupt:
    print("\nExiting experiment...")

finally:
    # Stop video recording
    if experiment_started:
        experiment_started = False
    if save_video:
        video_thread.join()  # video stops properly
        video_out.release()
    else:
        feed_thread.join()  # feed stops properly
    print("Video saved:", video_filename)

    camera.release()
    cv2.destroyAllWindows()
    GPIO.cleanup()
    arduino.close()
    gas_analyzer.close()
    print("GPIO cleaned up, devices closed.")
