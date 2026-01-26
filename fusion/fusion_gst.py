#!/usr/bin/env python3
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

import numpy as np
import cv2
import time

Gst.init(None)

# =========================
# GStreamer appsink wrapper
# =========================
class ShmAppSink:
    def __init__(self, pipeline_str, width, height, channels):
        self.width = width
        self.height = height
        self.channels = channels
        self.frame = None

        self.pipeline = Gst.parse_launch(pipeline_str)
        self.appsink = self.pipeline.get_by_name("appsink0")
        self.appsink.set_property("emit-signals", True)
        self.appsink.connect("new-sample", self.on_sample)

        self.pipeline.set_state(Gst.State.PLAYING)

    def on_sample(self, sink):
        sample = sink.emit("pull-sample")
        buf = sample.get_buffer()
        caps = sample.get_caps()
        data = buf.extract_dup(0, buf.get_size())

        arr = np.frombuffer(data, dtype=np.uint8)
        if self.channels == 1:
            arr = arr.reshape((self.height, self.width))
        else:
            arr = arr.reshape((self.height, self.width, self.channels))

        self.frame = arr
        return Gst.FlowReturn.OK


# =========================
# Pipelines
# =========================
PIPE_LEFT = (
    "shmsrc socket-path=/tmp/left_1280x800.sock is-live=true do-timestamp=true ! "
    "video/x-raw,format=RGB,width=1280,height=800,framerate=30/1 ! "
    "appsink name=appsink0 sync=false drop=true max-buffers=1"
)

PIPE_RIGHT = (
    "shmsrc socket-path=/tmp/right_1280x800.sock is-live=true do-timestamp=true ! "
    "video/x-raw,format=RGB,width=1280,height=800,framerate=30/1 ! "
    "appsink name=appsink0 sync=false drop=true max-buffers=1"
)

PIPE_DEPTH = (
    "shmsrc socket-path=/tmp/scdepth.sock is-live=true do-timestamp=true ! "
    "video/x-raw,format=GRAY8,width=320,height=256,framerate=30/1 ! "
    "appsink name=appsink0 sync=false drop=true max-buffers=1"
)

# =========================
# Stereo
# =========================
sgbm = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=128,
    blockSize=5,
    P1=8*3*5*5,
    P2=32*3*5*5,
    uniquenessRatio=10,
    speckleWindowSize=50,
    speckleRange=2,
    disp12MaxDiff=1,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
)

# =========================
# Main
# =========================
def main():
    left  = ShmAppSink(PIPE_LEFT, 1280, 800, 3)
    right = ShmAppSink(PIPE_RIGHT, 1280, 800, 3)
    depth = ShmAppSink(PIPE_DEPTH, 320, 256, 1)

    print("[INFO] Waiting for frames...")
    while left.frame is None or right.frame is None or depth.frame is None:
        time.sleep(0.01)

    print("[INFO] Frames incoming")

    while True:
        L = left.frame
        R = right.frame
        D = depth.frame

        # stereo disparity
        grayL = cv2.cvtColor(L, cv2.COLOR_RGB2GRAY)
        grayR = cv2.cvtColor(R, cv2.COLOR_RGB2GRAY)
        disp = sgbm.compute(grayL, grayR).astype(np.float32) / 16.0
        disp[disp <= 0] = np.nan

        # AI depth (near = dark)
        ai = D.astype(np.float32) / 255.0
        ai_up = cv2.resize(ai, (1280, 800))
        risk = 1.0 - ai_up

        # ROI (중앙)
        h, w = disp.shape
        roi = disp[int(0.4*h):int(0.7*h), int(0.4*w):int(0.6*w)]
        min_disp = np.nanpercentile(roi, 10)

        print(f"[ROI] disp10={min_disp:.2f}  ai_risk90={np.percentile(risk,90):.2f}", end="\r")

        # Debug view
        disp_vis = np.nan_to_num(disp, nan=0)
        disp_vis = (disp_vis / np.nanmax(disp_vis) * 255).astype(np.uint8)

        cv2.imshow("disp", disp_vis)
        cv2.imshow("ai_depth", (ai_up*255).astype(np.uint8))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
