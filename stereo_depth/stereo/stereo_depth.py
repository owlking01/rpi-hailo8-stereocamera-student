import numpy as np
import cv2


def load_calib(npz_path: str):
    """
    stereo_calib_result.npz 안에 들어있는 키 이름이
    너 코드 버전에 따라 조금 다를 수 있어서, 최대한 유연하게 읽는다.
    """
    data = np.load(npz_path)

    def pick(*names):
        for n in names:
            if n in data:
                return data[n]
        return None

    K1 = pick("K1", "mtxL", "camera_matrix_left", "cameraMatrix1")
    D1 = pick("D1", "distL", "dist_coeffs_left", "distCoeffs1")
    K2 = pick("K2", "mtxR", "camera_matrix_right", "cameraMatrix2")
    D2 = pick("D2", "distR", "dist_coeffs_right", "distCoeffs2")

    R  = pick("R", "R_lr", "R1to2")
    T  = pick("T", "T_lr", "T1to2")

    R1 = pick("R1")
    R2 = pick("R2")
    P1 = pick("P1")
    P2 = pick("P2")
    Q  = pick("Q")

    image_size = pick("image_size", "img_size", "size")
    if image_size is None:
        # fallback: 너가 지금 쓰는 기본
        image_size = np.array([640, 480])

    image_size = tuple(map(int, image_size.tolist()))

    if any(x is None for x in [K1, D1, K2, D2, R, T]):
        missing = [n for n, x in [("K1",K1),("D1",D1),("K2",K2),("D2",D2),("R",R),("T",T)] if x is None]
        raise ValueError(f"NPZ missing keys: {missing}. npz 안 키 이름을 보여주면 맞춰줄게.")

    return {
        "K1": K1, "D1": D1, "K2": K2, "D2": D2,
        "R": R, "T": T,
        "R1": R1, "R2": R2, "P1": P1, "P2": P2, "Q": Q,
        "image_size": image_size
    }


def build_rectify_maps(calib: dict, alpha: float = 0.0):
    """
    alpha=0: 겹치는 영역 위주(검은 테두리 최소)
    alpha=1: 원본 최대 유지(검은 테두리 증가)
    """
    K1, D1, K2, D2 = calib["K1"], calib["D1"], calib["K2"], calib["D2"]
    R, T = calib["R"], calib["T"]
    w, h = calib["image_size"]

    R1 = calib.get("R1")
    R2 = calib.get("R2")
    P1 = calib.get("P1")
    P2 = calib.get("P2")
    Q  = calib.get("Q")

    if any(x is None for x in [R1, R2, P1, P2, Q]):
        R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
            K1, D1, K2, D2, (w, h), R, T,
            flags=cv2.CALIB_ZERO_DISPARITY, alpha=alpha
        )

    map1x, map1y = cv2.initUndistortRectifyMap(K1, D1, R1, P1, (w, h), cv2.CV_16SC2)
    map2x, map2y = cv2.initUndistortRectifyMap(K2, D2, R2, P2, (w, h), cv2.CV_16SC2)

    return (map1x, map1y, map2x, map2y, Q)


def make_sgbm(min_disp=0, num_disp=128, block_size=7):
    """
    num_disp는 16의 배수여야 함.
    640x480에서 무난한 기본값.
    """
    if num_disp % 16 != 0:
        num_disp = (num_disp // 16 + 1) * 16

    sgbm = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=block_size,
        P1=8 * 3 * block_size * block_size,
        P2=32 * 3 * block_size * block_size,
        disp12MaxDiff=1,
        uniquenessRatio=8,
        speckleWindowSize=80,
        speckleRange=2,
        preFilterCap=31,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )
    return sgbm


def compute_disparity(sgbm, left_rect, right_rect):
    grayL = cv2.cvtColor(left_rect, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(right_rect, cv2.COLOR_BGR2GRAY)
    disp = sgbm.compute(grayL, grayR).astype(np.float32) / 16.0
    return disp


def depth_from_disparity_Q(disp, Q):
    """
    Q로 reproject -> Z(m) 뽑기
    (Z의 단위는 캘리브레이션 때 baseline 단위와 K 단위에 종속)
    """
    points_3d = cv2.reprojectImageTo3D(disp, Q)
    Z = points_3d[..., 2]
    return Z
