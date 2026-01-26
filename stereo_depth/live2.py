import cv2
import numpy as np
from .stereo.stereo_depth import (
    load_calib, build_rectify_maps, make_sgbm,
    compute_disparity, depth_from_disparity_Q
)
from .depth.scdepth_reader import open_scdepth_shm, scdepth_risk_from_colormap
from .fusion.depth_fusion import stereo_confidence, fuse_depth




def open_bayer_cam(dev="/dev/video0", w=640, h=480, fps=30):
    """
    oCam-1CGN-U-T2 (GRBG)용: GStreamer로 bayer2rgb 후 appsink
    """
    pipe = (
        f"v4l2src device={dev} io-mode=2 ! "
        f"video/x-bayer,format=grbg,width={w},height={h},framerate={fps}/1 ! "
        f"bayer2rgb ! videoconvert ! video/x-raw,format=BGR ! "
        f"appsink drop=true max-buffers=1 sync=false"
    )
    cap = cv2.VideoCapture(pipe, cv2.CAP_GSTREAMER)
    return cap


def main():
    # ===== 경로 (너 상황 고정) =====
    NPZ = "/home/kim/stereo_depth/calib/stereo_calib_result.npz"

    # ===== 카메라 매핑 (너가 말한 기준) =====
    # left_cam = video0, right_cam = video2  (필요하면 여기만 바꾸면 됨)
    left_dev  = "/dev/video0"
    right_dev = "/dev/video2"

    # ===== 오픈 =====
    capL = open_bayer_cam(left_dev)
    capR = open_bayer_cam(right_dev)
    if not capL.isOpened() or not capR.isOpened():
        raise RuntimeError("카메라 open 실패. /dev/video0,/dev/video2 잡히는지 확인.")

    capD = open_scdepth_shm("/tmp/scdepth.sock")
    if not capD.isOpened():
        print("[WARN] scdepth shm을 못 열었음. (터미널에서 depth gst-launch 먼저 켰는지 확인)")
        capD = None

    # ===== Stereo 준비 =====
    calib = load_calib(NPZ)
    map1x, map1y, map2x, map2y, Q = build_rectify_maps(calib, alpha=0.0)
    sgbm = make_sgbm(min_disp=0, num_disp=128, block_size=7)

    while True:
        okL, frameL = capL.read()
        okR, frameR = capR.read()
        if not okL or not okR:
            print("frame read 실패")
            break

        # rectify
        rectL = cv2.remap(frameL, map1x, map1y, cv2.INTER_LINEAR)
        rectR = cv2.remap(frameR, map2x, map2y, cv2.INTER_LINEAR)

        # disparity
        disp = compute_disparity(sgbm, rectL, rectR)
        valid = stereo_confidence(disp, min_disp=1.0, max_disp=200.0)

        # Z
        Z = depth_from_disparity_Q(disp, Q)  # unit은 캘리브레이션 설정에 종속

        # scdepth risk read
        if capD is not None:
            okD, depth_bgr = capD.read()
            if okD:
                risk_sc = scdepth_risk_from_colormap(depth_bgr)
                # scdepth는 320x256, stereo는 640x480 -> 리사이즈
                risk_sc = cv2.resize(risk_sc, (rectL.shape[1], rectL.shape[0]), interpolation=cv2.INTER_LINEAR)
            else:
                risk_sc = np.zeros((rectL.shape[0], rectL.shape[1]), dtype=np.float32)
        else:
            risk_sc = np.zeros((rectL.shape[0], rectL.shape[1]), dtype=np.float32)

        fused = fuse_depth(Z, valid, risk_sc, z_floor=0.2, z_ceiling=8.0)  # 0~1 위험도

        # 시각화: fused(0~1) -> heatmap
        fused_u8 = (np.clip(fused, 0, 1) * 255).astype(np.uint8)
        fused_color = cv2.applyColorMap(fused_u8, cv2.COLORMAP_JET)

        # 보기 좋게: rectified 좌/우 + fused
        top = np.hstack([rectL, rectR])
        fused_small = cv2.resize(fused_color, (top.shape[1], top.shape[0]))
        view = np.vstack([top, fused_small])

        cv2.imshow("Stereo(L/R) + Fused Risk", view)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord("q"):
            break

    capL.release()
    capR.release()
    if capD is not None:
        capD.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
    
