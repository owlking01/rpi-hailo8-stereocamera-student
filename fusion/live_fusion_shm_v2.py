#!/usr/bin/env python3
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst

import cv2
import numpy as np
import time
import argparse

Gst.init(None)

# =============================
# GStreamer appsink helper
# =============================
class ShmSink:
    def __init__(self, pipe, shape, name="sink"):
        self.frame = None
        self.shape = shape
        self.pipeline = Gst.parse_launch(pipe)
        self.sink = self.pipeline.get_by_name(name)
        self.sink.set_property("emit-signals", True)
        self.sink.connect("new-sample", self.on_sample)
        self.pipeline.set_state(Gst.State.PLAYING)

    def on_sample(self, sink):
        sample = sink.emit("pull-sample")
        buf = sample.get_buffer()
        data = buf.extract_dup(0, buf.get_size())
        arr = np.frombuffer(data, dtype=np.uint8).reshape(self.shape)
        self.frame = arr
        return Gst.FlowReturn.OK


# =============================
# Stereo calib: build PROC rectification maps + Q (recommended)
# =============================
def scale_K(K, sx, sy):
    K = K.astype(np.float64).copy()
    K[0,0] *= sx
    K[1,1] *= sy
    K[0,2] *= sx
    K[1,2] *= sy
    return K

def build_proc_rectify(npz_path, proc_size):
    """
    Uses K1,D1,K2,D2,R,T from npz.
    Recomputes stereoRectify at PROC size => consistent Q for PROC disparity.
    Then builds initUndistortRectifyMap for PROC.
    """
    d = np.load(npz_path, allow_pickle=True)
    keys = list(d.keys())
    print(f"[NPZ] {npz_path}")
    print(f"[NPZ] Keys: {keys}")

    K1 = d["K1"].astype(np.float64)
    D1 = d["D1"].astype(np.float64)
    K2 = d["K2"].astype(np.float64)
    D2 = d["D2"].astype(np.float64)
    R  = d["R"].astype(np.float64)
    T  = d["T"].astype(np.float64).reshape(3,1)

    # original image size
    if "image_size" in d:
        # could be (w,h) or (h,w). We'll infer.
        isz = d["image_size"]
        isz = tuple(int(x) for x in isz)
        if isz[0] > isz[1]:
            cap_w, cap_h = isz[0], isz[1]
        else:
            cap_h, cap_w = isz[0], isz[1]
    else:
        # fallback: assume 1280x800
        cap_w, cap_h = 1280, 800

    proc_w, proc_h = proc_size
    sx = proc_w / float(cap_w)
    sy = proc_h / float(cap_h)

    K1s = scale_K(K1, sx, sy)
    K2s = scale_K(K2, sx, sy)

    # Recompute rectify for PROC
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        K1s, D1, K2s, D2, (proc_w, proc_h), R, T,
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=0
    )

    map1x, map1y = cv2.initUndistortRectifyMap(
        K1s, D1, R1, P1, (proc_w, proc_h), cv2.CV_32FC1
    )
    map2x, map2y = cv2.initUndistortRectifyMap(
        K2s, D2, R2, P2, (proc_w, proc_h), cv2.CV_32FC1
    )

    # Baseline magnitude (for sanity print)
    baseline = float(np.linalg.norm(T))
    print(f"[CALIB] cap_size≈({cap_w},{cap_h})  proc_size=({proc_w},{proc_h})  baseline(norm T)={baseline:.4f}")
    print("[CALIB] Q ready for PROC disparity -> Z (same units as T).")
    return (map1x, map1y, map2x, map2y, Q)


def remap_pair(Lrgb, Rrgb, maps):
    map1x, map1y, map2x, map2y, Q = maps
    # OpenCV remap expects BGR usually, but maps work regardless; we only need grayscale later.
    Lr = cv2.remap(Lrgb, map1x, map1y, cv2.INTER_LINEAR)
    Rr = cv2.remap(Rrgb, map2x, map2y, cv2.INTER_LINEAR)
    return Lr, Rr


# =============================
# Confidence map (cheap)
# =============================
def confidence_from_grad(valid_mask, gray):
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx*gx + gy*gy)
    p95 = np.percentile(grad, 95) + 1e-6
    w = np.clip(grad / p95, 0.0, 1.0) * valid_mask.astype(np.float32)
    w = cv2.GaussianBlur(w, (5,5), 0)
    return np.clip(w, 0.0, 1.0)


# =============================
# Main
# =============================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="/home/kim/stereo_calib/stereo_calib_result.npz")
    ap.add_argument("--compute_every", type=int, default=2, help="Run SGBM every N frames (speed). 1=every frame")
    ap.add_argument("--show", action="store_true", help="Show mosaic debug window")
    args = ap.parse_args()

    # sizes
    CAP_W, CAP_H = 1280, 800
    PROC_W, PROC_H = 640, 480

    # ROI (PROC 기준)
    ROI_X0, ROI_X1 = int(0.40*PROC_W), int(0.60*PROC_W)
    ROI_Y0, ROI_Y1 = int(0.45*PROC_H), int(0.75*PROC_H)

    # thresholds
    EMA_ALPHA = 0.2
    RISK_TH = 0.65   # scdepth 위험 임계 (0~1)
    Z_DANGER = 1.2   # m (이 값은 네 환경에 맞춰 조정)

    # shm pipelines
    PIPE_L = (
        "shmsrc socket-path=/tmp/left_1280x800.sock is-live=true do-timestamp=true ! "
        "video/x-raw,format=RGB,width=1280,height=800,framerate=30/1 ! "
        "appsink name=sink drop=true max-buffers=1 sync=false"
    )
    PIPE_R = (
        "shmsrc socket-path=/tmp/right_1280x800.sock is-live=true do-timestamp=true ! "
        "video/x-raw,format=RGB,width=1280,height=800,framerate=30/1 ! "
        "appsink name=sink drop=true max-buffers=1 sync=false"
    )
    PIPE_D = (
        "shmsrc socket-path=/tmp/scdepth.sock is-live=true do-timestamp=true ! "
        "video/x-raw,format=GRAY8,width=320,height=256,framerate=30/1 ! "
        "appsink name=sink drop=true max-buffers=1 sync=false"
    )

    # start sinks
    left  = ShmSink(PIPE_L, (CAP_H, CAP_W, 3))
    right = ShmSink(PIPE_R, (CAP_H, CAP_W, 3))
    depth = ShmSink(PIPE_D, (256, 320))

    print("[INFO] Waiting for frames...")
    while left.frame is None or right.frame is None or depth.frame is None:
        time.sleep(0.01)
    print("[INFO] Frames OK")

    # build PROC rectify + Q (핵심)
    maps = build_proc_rectify(args.npz, (PROC_W, PROC_H))
    Q = maps[4]

    # SGBM tuned for PROC speed (live.py 느낌)
    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=96,   # 96 or 112 정도가 RPi에 현실적
        blockSize=5,
        P1=8 * 1 * 5 * 5,
        P2=32 * 1 * 5 * 5,
        uniquenessRatio=10,
        speckleWindowSize=60,
        speckleRange=2,
        disp12MaxDiff=1,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )

    z_ema = None
    last_disp = None
    last_Z = None
    last_w = None

    frame_i = 0
    t0 = time.time()
    fps_cnt = 0

    print("[RUN] q to quit (if --show).")

    while True:
        frame_i += 1
        fps_cnt += 1

        # 1) 가져오기 + PROC 다운스케일
        Lc = cv2.resize(left.frame,  (PROC_W, PROC_H), interpolation=cv2.INTER_AREA)
        Rc = cv2.resize(right.frame, (PROC_W, PROC_H), interpolation=cv2.INTER_AREA)

        # 2) rectification (PROC)
        Lr, Rr = remap_pair(Lc, Rc, maps)

        # 3) stereo 계산은 매 N프레임마다 (속도 핵심)
        do_compute = (frame_i % max(1, args.compute_every) == 0)

        if do_compute:
            gL = cv2.cvtColor(Lr, cv2.COLOR_RGB2GRAY)
            gR = cv2.cvtColor(Rr, cv2.COLOR_RGB2GRAY)

            disp = stereo.compute(gL, gR).astype(np.float32) / 16.0
            valid = (disp > 0.0) & np.isfinite(disp)
            disp[~valid] = np.nan

            # Q 기반 3D reproject => Z
            disp0 = np.nan_to_num(disp, nan=0.0).astype(np.float32)
            pts = cv2.reprojectImageTo3D(disp0, Q)
            Z = pts[:,:,2].astype(np.float32)
            Z[~valid] = np.nan

            # confidence (cheap)
            w = confidence_from_grad(valid, gL)

            last_disp, last_Z, last_w = disp, Z, w
        else:
            disp, Z, w = last_disp, last_Z, last_w

        # 4) scdepth risk (PROC로 리사이즈 + ROI)
        ai = depth.frame.astype(np.float32) / 255.0      # bright=far
        ai = cv2.resize(ai, (PROC_W, PROC_H), interpolation=cv2.INTER_LINEAR)
        risk = 1.0 - ai                                  # near=high

        risk_roi = risk[ROI_Y0:ROI_Y1, ROI_X0:ROI_X1]
        risk90 = float(np.percentile(risk_roi, 90))

        # 5) ROI 거리 (m) : Z의 10퍼센타일(가까운 쪽)
        if Z is None:
            z10 = np.nan
            conf = 0.0
        else:
            Z_roi = Z[ROI_Y0:ROI_Y1, ROI_X0:ROI_X1]
            z10 = float(np.nanpercentile(Z_roi, 10)) if np.isfinite(Z_roi).any() else np.nan
            conf = float(np.nanmean(w[ROI_Y0:ROI_Y1, ROI_X0:ROI_X1])) if w is not None else 0.0

        # EMA (거리 안정화)
        if np.isfinite(z10):
            if z_ema is None:
                z_ema = z10
            else:
                z_ema = (1.0-0.2)*z_ema + 0.2*z10

        # 6) Danger gate (보수적)
        danger = (risk90 > RISK_TH) or (z_ema is not None and z_ema < Z_DANGER) or (conf < 0.15)

        # FPS print (2초마다)
        now = time.time()
        if now - t0 > 2.0:
            fps = fps_cnt / (now - t0)
            print(f"\n[FPS] ~{fps:.1f}  compute_every={args.compute_every}")
            t0 = now
            fps_cnt = 0

        # 7) 모자이크 디버그 (1창) — 원하면 --show
        if args.show:
            # base view
            gL_vis = cv2.cvtColor(cv2.cvtColor(Lr, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2BGR)
            cv2.rectangle(gL_vis, (ROI_X0, ROI_Y0), (ROI_X1, ROI_Y1), (0,255,0), 2)

            # disp vis
            if disp is not None:
                dv = np.nan_to_num(disp, nan=0.0)
                mx = np.nanpercentile(dv, 99) + 1e-6
                dv = np.clip(dv/mx, 0, 1)
                dv = (dv*255).astype(np.uint8)
                dv = cv2.cvtColor(dv, cv2.COLOR_GRAY2BGR)
            else:
                dv = np.zeros_like(gL_vis)

            # ai/risk vis
            ai8 = (ai*255).astype(np.uint8)
            rv8 = (risk*255).astype(np.uint8)
            aiB = cv2.cvtColor(ai8, cv2.COLOR_GRAY2BGR)
            rvB = cv2.cvtColor(rv8, cv2.COLOR_GRAY2BGR)

            # conf vis
            if w is not None:
                w8 = (np.clip(w,0,1)*255).astype(np.uint8)
                wB = cv2.cvtColor(w8, cv2.COLOR_GRAY2BGR)
            else:
                wB = np.zeros_like(gL_vis)

            # shrink tiles to reduce GUI cost
            def small(x):
                return cv2.resize(x, (PROC_W//2, PROC_H//2))

            row1 = np.hstack([small(gL_vis), small(dv)])
            row2 = np.hstack([small(aiB),    small(wB)])
            mosaic = np.vstack([row1, row2])

            txt = f"Z10={z10:.2f}m  Z_ema={z_ema:.2f}m  risk90={risk90:.2f}  conf={conf:.2f}"
            color = (0,0,255) if danger else (0,255,0)
            cv2.putText(mosaic, txt, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            if danger:
                cv2.putText(mosaic, "[DANGER]", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 3)

            cv2.imshow("fusion_mosaic", mosaic)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            # 터미널 상태 출력(너무 spam 안 되게)
            if frame_i % 10 == 0:
                print(f"[STAT] Z_ema={z_ema if z_ema is not None else np.nan:.2f}m  risk90={risk90:.2f}  conf={conf:.2f}  {'DANGER' if danger else 'OK'}", end="\r")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
