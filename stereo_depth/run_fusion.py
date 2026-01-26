import numpy as np
import cv2

from stereo_depth import StereoDepth
from depth_fusion import DepthFusion

stereo = StereoDepth("/home/kim/stereo_calib/stereo_calib_result.npz")
fusion = DepthFusion()

# 임시 테스트용 (나중에 live 코드로 교체)
disparity = np.load("disparity.npy")
scdepth = np.load("scdepth.npy")

Z_stereo = stereo.disparity_to_depth(disparity)
Z_fused = fusion.fuse(Z_stereo, disparity, scdepth)

if Z_fused is None:
    print("Fusion failed")
else:
    vis = np.clip(Z_fused, 0, 3.0)
    vis = (vis / 3.0 * 255).astype(np.uint8)
    cv2.imshow("Z_fused", vis)
    cv2.waitKey(0)
