#!/usr/bin/env python3
import time, math, argparse
import numpy as np
import cv2

from pymavlink import mavutil
import hailo_platform as hpf

# ================= CONFIG =================
HEF_PATH = "/home/kim/zero_dce_pp.hef"
DEV = "/dev/video0"

CAP_W, CAP_H = 640, 480
NET_W, NET_H = 600, 400

# AR0144 + 600x400 기준 (FOV=91° 근사)
FX = 294.0
FY = 196.0

ROI_REL = (0.25, 0.75, 0.25, 0.75)

TAU = 0.25
STATIONARY_SPEED_PXS = 3.0


# ================= CAMERA =================
def open_cam(dev):
    cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError("Camera open failed")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_H)
    cap.set(cv2.CAP_PROP_FPS, 30)
    return cap


# ================= HAILO ZERO-DCE =================
class HailoZeroDCE:
    def __init__(self, hef_path):
        self.hef = hpf.HEF(hef_path)
        self.vdevice = hpf.VDevice()

        cfg = hpf.ConfigureParams.create_from_hef(
            self.hef, interface=hpf.HailoStreamInterface.PCIe
        )
        self.ng = self.vdevice.configure(self.hef, cfg)[0]
        self.ng_params = self.ng.create_params()

        self.in_params = hpf.InputVStreamParams.make_from_network_group(
            self.ng, quantized=True, format_type=hpf.FormatType.UINT8
        )
        self.out_params = hpf.OutputVStreamParams.make_from_network_group(
            self.ng, quantized=True, format_type=hpf.FormatType.UINT8
        )

        self.in_name = self.hef.get_input_vstream_infos()[0].name
        self.out_name = self.hef.get_output_vstream_infos()[0].name

    def __enter__(self):
        self.ctx_ng = self.ng.activate(self.ng_params)
        self.ctx_ng.__enter__()
        self.ctx_infer = hpf.InferVStreams(self.ng, self.in_params, self.out_params)
        self.ctx_infer.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.ctx_infer.__exit__(exc_type, exc, tb)
        self.ctx_ng.__exit__(exc_type, exc, tb)
        self.vdevice.release()

    def infer(self, rgb):
        inp = {self.in_name: np.expand_dims(rgb, 0)}
        out = self.ctx_infer.infer(inp)[self.out_name]
        return out[0].astype(np.uint8)


# ================= OPTICAL FLOW =================
class LKFlow:
    def __init__(self):
        self.prev_gray = None
        self.prev_pts = None
        self.prev_time = None
        self.vx_f = 0.0
        self.vy_f = 0.0
        self.ema_inited = False

    def _init_pts(self, gray):
        return cv2.goodFeaturesToTrack(
            gray, 300, 0.01, 8
        )

    def step(self, frame):

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        x0 = int(w * ROI_REL[0])
        x1 = int(w * ROI_REL[1])
        y0 = int(h * ROI_REL[2])
        y1 = int(h * ROI_REL[3])

        roi = gray[y0:y1, x0:x1]

        now = time.time()
        if self.prev_time is None:
            self.prev_time = now
            self.prev_gray = gray
            self.prev_pts = self._init_pts(roi)
            return frame, 0.0, 0.0, 0.0, False

        dt = now - self.prev_time
        self.prev_time = now

        dx = dy = 0.0
        r = 0.0
        valid = False

        if dt > 1e-3:

            prev_roi = self.prev_gray[y0:y1, x0:x1]

            if self.prev_pts is None or len(self.prev_pts) < 50:
                self.prev_pts = self._init_pts(prev_roi)

            if self.prev_pts is not None:
                nxt, st, _ = cv2.calcOpticalFlowPyrLK(
                    prev_roi, roi, self.prev_pts, None
                )

                st = st.reshape(-1)
                p0 = self.prev_pts.reshape(-1, 2)[st == 1]
                p1 = nxt.reshape(-1, 2)[st == 1]

                total = len(p0)

                if total > 10:
                    _, mask = cv2.estimateAffinePartial2D(
                        p0, p1, method=cv2.RANSAC
                    )

                    if mask is not None:
                        mask = mask.reshape(-1).astype(bool)
                        inliers = int(np.sum(mask))
                        r = inliers / max(1, total)

                        dp = (p1 - p0)[mask]
                        dx = float(np.median(dp[:, 0])) if len(dp) else 0.0
                        dy = float(np.median(dp[:, 1])) if len(dp) else 0.0

                        speed = math.hypot(dx/dt, dy/dt)

                        if r > 0.5 and speed > STATIONARY_SPEED_PXS:
                            valid = True
                        else:
                            dx = dy = 0.0

                        # EMA
                        alpha = 1 - math.exp(-dt/TAU)
                        if not self.ema_inited:
                            self.vx_f = dx/dt
                            self.vy_f = dy/dt
                            self.ema_inited = True
                        else:
                            self.vx_f = (1-alpha)*self.vx_f + alpha*(dx/dt)
                            self.vy_f = (1-alpha)*self.vy_f + alpha*(dy/dt)

                        for (a,b),(c,d),ok in zip(p0,p1,mask):
                            if ok:
                                cv2.line(frame,(int(a)+x0,int(b)+y0),
                                         (int(c)+x0,int(d)+y0),(0,255,0),1)

        self.prev_gray = gray
        self.prev_pts = self._init_pts(roi)

        rad_x = dx / FX
        rad_y = dy / FY

        cv2.rectangle(frame,(x0,y0),(x1,y1),(255,0,0),2)

        cv2.putText(frame,f"dx {dx:+.2f}px",(10,30),
                    cv2.FONT_HERSHEY_SIMPLEX,0.7,(200,200,200),2)

        cv2.putText(frame,f"flow_x {rad_x:+.4f}rad",(10,60),
                    cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,255),2)

        cv2.putText(frame,f"flow_y {rad_y:+.4f}rad",(10,95),
                    cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,255),2)

        cv2.putText(frame,f"r {r:.2f} valid {int(valid)}",(10,125),
                    cv2.FONT_HERSHEY_SIMPLEX,0.7,(200,200,200),2)

        return frame, rad_x, rad_y, r, valid


# ================= MAVLINK =================
def setup_mav(port, baud):
    m = mavutil.mavlink_connection(port, baud=baud)
    m.wait_heartbeat()
    return m


def send_flow(mav, rad_x, rad_y, dt, quality, distance_m=0.0):

    mav.mav.optical_flow_rad_send(
        int(time.time()*1e6),   # time_usec
        0,                     # sensor_id  <<--- 추가
        int(dt*1e6),           # integration_time_us
        float(rad_x),          # integrated_x
        float(rad_y),          # integrated_y
        0.0,                   # integrated_xgyro
        0.0,                   # integrated_ygyro
        0.0,                   # integrated_zgyro
        0,                     # temperature
        int(quality),          # quality
        0,                     # time_delta_distance_us
        float(distance_m)      # distance
    )




# ================= MAIN =================
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--mav_port", default="/dev/ttyACM0")
    args = parser.parse_args()

    cap = open_cam(DEV)
    mav = setup_mav(args.mav_port,115200)
    flow = LKFlow()

    with HailoZeroDCE(HEF_PATH) as ll:

        while True:

            ok, frame = cap.read()
            if not ok:
                continue

            frame_rs = cv2.resize(frame,(NET_W,NET_H))
            rgb = cv2.cvtColor(frame_rs,cv2.COLOR_BGR2RGB)

            enhanced = ll.infer(rgb)
            enhanced_bgr = cv2.cvtColor(enhanced,cv2.COLOR_RGB2BGR)

            vis, rad_x, rad_y, r, valid = flow.step(enhanced_bgr)

            quality = int(max(0,min(1,r))*255)

            if not valid:
                rad_x = rad_y = 0.0
                quality = 0

            send_flow(mav, rad_x, rad_y, 0.03, quality, 0.0)

            cv2.imshow("Hailo OF MAV",vis)

            if cv2.waitKey(1)&0xFF==ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
