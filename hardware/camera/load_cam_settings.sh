#!/bin/bash
#
V4L2_DEVICE="/dev/video0" # default usb video input, unless multiple
#
# set controls for camera
v4l2-ctl -d $V4L2_DEVICE --set-fmt-video=width=1280,height=1080,pixelformat=0
v4l2-ctl -d $V4L2_DEVICE --set-parm=30
v4l2-ctl -d $V4L2_DEVICE --set-ctrl brightness=0
v4l2-ctl -d $V4L2_DEVICE --set-ctrl contrast=32
v4l2-ctl -d $V4L2_DEVICE --set-ctrl saturation=0
v4l2-ctl -d $V4L2_DEVICE --set-ctrl gain=0
v4l2-ctl -d $V4L2_DEVICE --set-ctrl power_line_frequency=0
v4l2-ctl -d $V4L2_DEVICE --set-ctrl sharpness=10
v4l2-ctl -d $V4L2_DEVICE --set-ctrl white_balance_temperature=2800
v4l2-ctl -d $V4L2_DEVICE --set-ctrl backlight_compensation=0
v4l2-ctl -d $V4L2_DEVICE --set-ctrl auto_exposure=1
v4l2-ctl -d $V4L2_DEVICE --set-ctrl exposure_time_absolute=200
v4l2-ctl -d $V4L2_DEVICE --set-ctrl exposure_dynamic_framerate=0
#
echo "V4L2 settings loaded for $V4L2_DEVICE"
# if you need to run manually, use command "./load_cam_settings.sh" in terminal
