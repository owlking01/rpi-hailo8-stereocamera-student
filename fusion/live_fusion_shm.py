#!/usr/bin/env python3
import os
import time
import argparse
import numpy as np
import cv2

import gi
gi.require_version("Gst", "1.0")
gi.require_version("GstApp", "1.0")
from gi.repository import Gst, GstApp

Gst.init(None)


# =============================
# NPZ utils (same as your live.py)
# =============================
def load_npz(npz_path: str):
    d = np.load(npz_path, allow_pickle=True)
    need = ["K1", "D1", "K2", "D2", "R1", "R2", "P1", "P2", "T"]
    keys = list(d.keys())
    for k in need:
        if k not in d:
            raise RuntimeError(f"NPZ missing key: {k}\nkeys={keys}")
    return d


def scale_K(K, sx, sy):
    K2 = K.copy().astype(np.float64)
    K2[0, 0] *= sx  # fx
    K2[1, 1] *= sy  # fy
    K2[0, 2] *= sx  # cx
    K2[1, 2] *= sy  # cy
    return K2


def scale_P(P, sx, sy):
    P2 = P.copy().astype(np.float64)
    P2[0, 0] *= sx  # fx
    P2[1, 1] *= sy  # fy
    P2[0, 2] *= sx  # cx
    P2[1, 2] *= sy  # cy
    P2[0, 3] *= sx
    P2[1, 3] *= sy
    return P2


def build_maps_for_proc(d, cap_w, cap_h, proc_w, proc_h):
    # EXACTLY match your live.py
    sx = proc_w / float(cap_w)
    sy = proc_h / float(cap_h)

    K1 = np.array(d["K1"])
    D1 = np.array(d["D1"]).reshape(-1, 1)
    K2 = np.array(d["K2"])
    D2 = np.array(d["D2"]).reshape(-1, 1)

    R1 = np.array(d["R1"])
    R2 = np.array(d["R2"])
    P1 = np.array(d["P1"])
    P2 = np.array(d["P2"])

    K1s = scale_K(K1, sx, sy)
    K2s = scale_K(K2, sx, sy)
    P1s = scale_P(P1, sx, sy)
    P2s = scale_P(P2, sx, sy)

    size = (proc_w, proc_h)
    map1x, map1y = cv2.initUndistortRectifyMap(K1s, D1, R1, P1s, size, cv2.CV_16SC2)
    map2x, map2y = cv2.initUndistortRectifyMap(K2s, D2, R2, P2s, size, cv2.CV_16SC2)

    fx_proc = float(P1s[0, 0])
    return map1x, map1y, map2x, map2y, fx_proc


def build_metric(d):
    T = np.array(d["T"]).reshape(3)
    baseline_m = float(np.linalg.norm(T))
    cs = None
    if "circle_spacing" in d:
        try:
            cs = float(np.array(d["circle_spacing"]).reshape(()))
        except Exception:
            cs = None
    return baseline_m, cs


# =============================
# GStreamer shmsrc reader (NO OpenCV-GStreamer needed)
# =============================
class ShmAppSinkReader:
    """
    Pull raw frames from shmsrc via appsink.
    format: RGB or GRAY8
    """
    def __init__(self, socket_path, width, height, fps, fmt="RGB"):
        self.socket_path = socket_path
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.fmt = fmt  # "RGB" or "GRAY8"

        caps = f"video/x-raw,format={fmt},width={self.width},height={self.height},framerate={self.fps}/1"
        pipe_str = (
            f"shmsrc socket-path={socket_path} is-live=true do-timestamp=true ! "
            f"{caps} ! "
            f"queue leaky=downstream max-size-buffers=1 ! "
            f"appsink name=appsink emit-signals=false sync=false max-buffers=1 drop=true"
        )

        self.pipeline = Gst.parse_launch(pipe_str)
        self.appsink = self.pipeline.get_by_name("appsink")
        if self.appsink is None:
            raise RuntimeError("Failed to get appsink from pipeline")

        self.pipeline.set_state(Gst.State.PLAYING)

    def read(self, timeout_ms=200):
        sample = self.appsink.try_pull_sample(timeout_ms * Gst.MSECOND)
        if sample is None:
            return False, None

        buf = sample.get_buffer()
        caps = sample.get_caps()
        s = caps.get_structure(0)
        w = s.get_value("width")
        h = s.get_value("height")
        fmt = s.get_value("format")

        ok, mapinfo = buf.map(Gst.MapFlags.READ)
        if not ok:
            return False, None

        try:
            data = mapinfo.data
            if fmt == "RGB":
                arr = np.frombuffer(data, dtype=np.uint8).reshape((h, w, 3))
                frame = arr.copy()
            elif fmt == "GRAY8":
                arr = np.frombuffer(data, dtype=np.uint8).reshape((h, w))
                frame = arr.copy()
            else:
                return False, None
        finally:
            buf.unmap(mapinfo)

        return True, frame

    def release(self):
        try:
            self.pipeline.set_state(Gst.State.NULL)
        except Exception:
            pass


# =============================
# scdepth risk + stereo confidence
# =============================
def compute_scdepth_risk(depth_gray_u8, roi, invert_close=True):
    x0, y0, x1, y1 = roi
    h, w = depth_gray_u8.shape[:2]
    x0 = int(np.clip(x0, 0, w-1)); x1 = int(np.clip(x1, 0, w))
    y0 = int(np.clip(y0, 0, h-1)); y1 = int(np.clip(y1, 0, h))
    if x1 <= x0 or y1 <= y0:
        return 0.0

    patch = depth_gray_u8[y0:y1, x0:x1].astype(np.float32)
    if patch.size < 10:
        return 0.0

    # close = darker => invert
    if invert_close:
        closeness = (255.0 - patch) / 255.0
    else:
        closeness = patch / 255.0

    # worst-case-ish in ROI
    return float(np.percentile(closeness, 90))


def compute_confidence_from_edges(gray, roi):
    x0, y0, x1, y1 = roi
    h, w = gray.shape[:2]
    x0 = int(np.clip(x0, 0, w-1)); x1 = int(np.clip(x1, 0, w))
    y0 = int(np.clip(y0, 0, h-1)); y1 = int(np.clip(y1, 0, h))
    if x1 <= x0 or y1 <= y0:
        return 0.0

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    patch = mag[y0:y1, x0:x1]
    if patch.size < 10:
        return 0.0

    m = float(np.mean(patch))
    return float(np.clip(m / 40.0, 0.0, 1.0))


def fmt(v):
    return f"{v:.2f}" if (v is not None and np.isfinite(v)) else "NA"


def safe_bool(x):
    return bool(x) if x is not None else False


# =============================
# Main (with hysteresis gating)
# =============================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default=os.path.expanduser("~/stereo_calib/stereo_calib_result.npz"))

    ap.add_argument("--left_sock", default="/tmp/left_1280x800.sock")
    ap.add_argument("--right_sock", default="/tmp/right_1280x800.sock")
    ap.add_argument("--depth_sock", default="/tmp/scdepth.sock")

    ap.add_argument("--cap_w", type=int, default=1280)
    ap.add_argument("--cap_h", type=int, default=800)
    ap.add_argument("--proc_w", type=int, default=640)
    ap.add_argument("--proc_h", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30)

    ap.add_argument("--depth_w", type=int, default=320)
    ap.add_argument("--depth_h", type=int, default=256)

    # ROI ratios (same as your live.py)
    ap.add_argument("--roi_x0", type=float, default=0.30)
    ap.add_argument("--roi_x1", type=float, default=0.70)
    ap.add_argument("--roi_y0", type=float, default=0.55)
    ap.add_argument("--roi_y1", type=float, default=0.80)

    # smoothing / compute
    ap.add_argument("--ema_alpha", type=float, default=0.2)
    ap.add_argument("--compute_every", type=int, default=2)

    # --- Hysteresis thresholds ---
    # Stereo trust hysteresis
    ap.add_argument("--conf_off", type=float, default=0.22, help="If stereo trusted and conf < conf_off => stop trusting stereo")
    ap.add_argument("--conf_on", type=float, default=0.32, help="If stereo not trusted and conf > conf_on => trust stereo")

    # Danger hysteresis from scdepth risk
    ap.add_argument("--risk_on", type=float, default=0.88, help="If not danger and risk90 > risk_on => danger")
    ap.add_argument("--risk_off", type=float, default=0.78, help="If danger and risk90 < risk_off => clear danger (if no other danger reason)")

    # Optional stereo distance danger
    ap.add_argument("--z_close_on", type=float, default=1.0, help="If stereo trusted and Z_ema < z_close_on => danger")
    ap.add_argument("--z_close_off", type=float, default=1.2, help="If danger by stereo and Z_ema > z_close_off => clear (hysteresis)")

    ap.add_argument("--show", action="store_true")
    ap.add_argument("--log_csv", default="", help="Optional CSV log path, e.g. /home/kim/fusion/log.csv")
    args = ap.parse_args()

    # --- Load calibration ---
    d = load_npz(args.npz)
    baseline_m, cs = build_metric(d)
    mapLx, mapLy, mapRx, mapRy, fx = build_maps_for_proc(d, args.cap_w, args.cap_h, args.proc_w, args.proc_h)

    print("[NPZ]", args.npz)
    print(f"[metric] baseline={baseline_m:.4f} m (||T||)")
    if cs is not None:
        print(f"[metric] circle_spacing={cs:.6f} (NOT used)")
    print(f"[metric] fx(PROC)={fx:.2f} px  PROC={args.proc_w}x{args.proc_h}  CAP={args.cap_w}x{args.cap_h}")
    print(f"[shm] L={args.left_sock}  R={args.right_sock}  depth={args.depth_sock}")

    # --- Readers ---
    left = ShmAppSinkReader(args.left_sock, args.cap_w, args.cap_h, args.fps, fmt="RGB")
    right = ShmAppSinkReader(args.right_sock, args.cap_w, args.cap_h, args.fps, fmt="RGB")

    depth = None
    try:
        depth = ShmAppSinkReader(args.depth_sock, args.depth_w, args.depth_h, args.fps, fmt="GRAY8")
    except Exception as e:
        print("[WARN] scdepth reader disabled:", e)

    # --- StereoSGBM EXACT from your live.py ---
    min_disp = 0
    num_disp = 16 * 6  # 96
    block = 5
    sgbm = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=block,
        P1=8 * 1 * block * block,
        P2=32 * 1 * block * block,
        disp12MaxDiff=1,
        uniquenessRatio=8,
        speckleWindowSize=80,
        speckleRange=2,
        preFilterCap=31,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )

    # ROI in PROC
    ROI_X0 = int(args.proc_w * args.roi_x0)
    ROI_X1 = int(args.proc_w * args.roi_x1)
    ROI_Y0 = int(args.proc_h * args.roi_y0)
    ROI_Y1 = int(args.proc_h * args.roi_y1)
    roi_proc = (ROI_X0, ROI_Y0, ROI_X1, ROI_Y1)

    def roi_proc_to_depth():
        sx = args.depth_w / float(args.proc_w)
        sy = args.depth_h / float(args.proc_h)
        return (int(ROI_X0 * sx), int(ROI_Y0 * sy), int(ROI_X1 * sx), int(ROI_Y1 * sy))

    # Logging
    csv_f = None
    if args.log_csv:
        os.makedirs(os.path.dirname(args.log_csv), exist_ok=True)
        csv_f = open(args.log_csv, "w", buffering=1)
        csv_f.write("t,mode,danger,reason,conf,risk90,stereo_valid,valid_cnt,d_med,z_now,z_ema\n")
        print("[LOG] CSV:", args.log_csv)

    # State vars
    z_ema = None
    stereo_trusted = True  # start trusted; hysteresis will settle
    danger = False
    danger_by = ""  # "RISK" or "STEREO" or "BOTH"

    idx = 0
    frames = 0
    last_t = time.time()
    fps_text = 0.0

    last_disp = None
    last_risk90 = 0.0
    last_conf = 0.0
    last_dmed = None
    last_znow = None
    last_valid_cnt = 0
    last_stereo_valid = False

    if args.show:
        cv2.namedWindow("fusion_mosaic", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("fusion_mosaic", args.proc_w * 2, args.proc_h * 2)

    print("[RUN] q to quit (if --show).")
    print(f"[GATE] conf_off={args.conf_off:.2f} conf_on={args.conf_on:.2f} | risk_on={args.risk_on:.2f} risk_off={args.risk_off:.2f} | z_on={args.z_close_on:.2f} z_off={args.z_close_off:.2f}")

    while True:
        idx += 1

        okL, rgbL = left.read(timeout_ms=200)
        okR, rgbR = right.read(timeout_ms=200)
        if not okL or not okR:
            time.sleep(0.005)
            continue

        # RGB -> BGR for OpenCV
        bgrL_cap = rgbL[:, :, ::-1]
        bgrR_cap = rgbR[:, :, ::-1]

        # Match live.py: CAP -> PROC resize, then PROC rectify
        bgrL = cv2.resize(bgrL_cap, (args.proc_w, args.proc_h), interpolation=cv2.INTER_AREA)
        bgrR = cv2.resize(bgrR_cap, (args.proc_w, args.proc_h), interpolation=cv2.INTER_AREA)

        rectL = cv2.remap(bgrL, mapLx, mapLy, cv2.INTER_LINEAR)
        rectR = cv2.remap(bgrR, mapRx, mapRy, cv2.INTER_LINEAR)

        gL = cv2.cvtColor(rectL, cv2.COLOR_BGR2GRAY)
        gR = cv2.cvtColor(rectR, cv2.COLOR_BGR2GRAY)

        # Depth (optional)
        depth_gray = None
        if depth is not None:
            okD, dimg = depth.read(timeout_ms=5)
            if okD:
                depth_gray = dimg

        # Always compute conf from edges (cheap)
        conf = compute_confidence_from_edges(gL, roi_proc)

        # Always compute risk if depth exists (cheap)
        risk90 = 0.0
        if depth_gray is not None:
            risk90 = compute_scdepth_risk(depth_gray, roi_proc_to_depth(), invert_close=True)

        # --- Stereo compute (expensive) every N frames ---
        if (idx % max(1, args.compute_every)) == 0 or last_disp is None:
            disp = sgbm.compute(gL, gR).astype(np.float32) / 16.0
            disp[disp < 0] = 0

            roi = disp[ROI_Y0:ROI_Y1, ROI_X0:ROI_X1]
            valid = (roi > 1.0) & (roi < (num_disp - 1)) & np.isfinite(roi)
            valid_cnt = int(np.count_nonzero(valid))

            z_now = None
            d_med = None
            stereo_valid = False
            if valid_cnt > 80:
                d_med = float(np.median(roi[valid]))
                z_now = (fx * baseline_m) / max(d_med, 1e-6)
                stereo_valid = True

                if z_ema is None:
                    z_ema = z_now
                else:
                    z_ema = (1.0 - args.ema_alpha) * z_ema + args.ema_alpha * z_now

            # Save last
            last_disp = disp
            last_dmed = d_med
            last_znow = z_now
            last_valid_cnt = valid_cnt
            last_stereo_valid = stereo_valid

        # --- Stereo trust hysteresis ---
        # If currently trusted, drop trust when conf too low OR stereo invalid
        if stereo_trusted:
            if (conf < args.conf_off) or (not last_stereo_valid):
                stereo_trusted = False
        else:
            # If currently not trusted, re-enable when conf is high AND stereo is valid
            if (conf > args.conf_on) and last_stereo_valid:
                stereo_trusted = True

        # --- Danger hysteresis ---
        # Danger reasons:
        #   - RISK: scdepth says close (risk90)
        #   - STEREO: if stereo trusted and z_ema indicates too close
        reason_parts = []

        # 1) risk-based
        risk_danger = False
        if not danger:
            if risk90 > args.risk_on:
                risk_danger = True
        else:
            # when already in danger, keep it until risk goes below risk_off
            if risk90 > args.risk_off:
                risk_danger = True

        # 2) stereo-distance-based (only if stereo trusted and we have z_ema)
        stereo_danger = False
        if stereo_trusted and (z_ema is not None) and np.isfinite(z_ema):
            if not danger:
                if z_ema < args.z_close_on:
                    stereo_danger = True
            else:
                # when already in danger, keep until z > z_close_off
                if z_ema < args.z_close_off:
                    stereo_danger = True

        # Combine
        new_danger = (risk_danger or stereo_danger)

        if new_danger:
            if risk_danger:
                reason_parts.append("RISK")
            if stereo_danger:
                reason_parts.append("Z_CLOSE")
            danger_by = "+".join(reason_parts) if reason_parts else "UNKNOWN"
        else:
            danger_by = ""

        danger = new_danger

        # Mode text
        mode = "STEREO" if stereo_trusted else "SCDEPTH"
        # Provide reason for mode
        mode_reason = []
        if not stereo_trusted:
            if conf < args.conf_off:
                mode_reason.append("conf_low")
            if not last_stereo_valid:
                mode_reason.append("stereo_invalid")
        else:
            if conf > args.conf_on and last_stereo_valid:
                mode_reason.append("conf_ok")

        reason = f"mode={'/'.join(mode_reason) if mode_reason else 'ok'}; danger={danger_by if danger else 'none'}"

        # FPS
        frames += 1
        now = time.time()
        if now - last_t >= 1.0:
            fps_text = frames / (now - last_t)
            frames = 0
            last_t = now

        # CSV log
        if csv_f is not None:
            csv_f.write(
                f"{now:.3f},{mode},{int(danger)},{danger_by},{conf:.3f},{risk90:.3f},{int(last_stereo_valid)},{last_valid_cnt},"
                f"{'' if last_dmed is None else f'{last_dmed:.3f}'},"
                f"{'' if last_znow is None else f'{last_znow:.3f}'},"
                f"{'' if z_ema is None else f'{z_ema:.3f}'}\n"
            )

        if args.show:
            # TL: rectL + ROI
            tl = rectL.copy()
            cv2.rectangle(tl, (ROI_X0, ROI_Y0), (ROI_X1, ROI_Y1), (0, 255, 0), 2)

            # TR: disparity viz
            disp_vis = cv2.normalize(last_disp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            tr = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)

            # BL: scdepth
            if depth_gray is None:
                bl = np.zeros((args.proc_h, args.proc_w, 3), dtype=np.uint8)
                cv2.putText(bl, "scdepth: NA", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            else:
                bl = cv2.resize(depth_gray, (args.proc_w, args.proc_h), interpolation=cv2.INTER_NEAREST)
                bl = cv2.cvtColor(bl, cv2.COLOR_GRAY2BGR)

            # BR: edge mag (confidence viz)
            gx = cv2.Sobel(gL, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(gL, cv2.CV_32F, 0, 1, ksize=3)
            mag = cv2.magnitude(gx, gy)
            mag_u8 = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            br = cv2.applyColorMap(mag_u8, cv2.COLORMAP_JET)
            cv2.rectangle(br, (ROI_X0, ROI_Y0), (ROI_X1, ROI_Y1), (255, 255, 255), 2)

            vis = np.vstack([np.hstack([tl, tr]), np.hstack([bl, br])])

            # Overlay: mode/danger big
            header = f"MODE={mode}  stereo_valid={int(last_stereo_valid)}  valid_cnt={last_valid_cnt}  FPS={fps_text:.1f}"
            stats = f"conf={conf:.2f}  risk90={risk90:.2f}  d_med={fmt(last_dmed)}px  Z_now={fmt(last_znow)}m  Z_ema={fmt(z_ema)}m"
            danger_txt = f"DANGER={'YES' if danger else 'NO'}  by={danger_by if danger else '-'}"

            # top bar background
            cv2.rectangle(vis, (0, 0), (vis.shape[1], 85), (0, 0, 0), -1)
            cv2.putText(vis, header, (20, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(vis, stats, (20, 56),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            color = (0, 0, 255) if danger else (0, 255, 0)
            cv2.putText(vis, danger_txt, (20, 84),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            # reason small footer
            cv2.rectangle(vis, (0, vis.shape[0]-28), (vis.shape[1], vis.shape[0]), (0, 0, 0), -1)
            cv2.putText(vis, reason, (20, vis.shape[0]-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

            cv2.imshow("fusion_mosaic", vis)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    left.release()
    right.release()
    if depth is not None:
        depth.release()
    if csv_f is not None:
        csv_f.close()
    if args.show:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
