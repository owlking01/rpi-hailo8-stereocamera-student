import numpy as np


def stereo_confidence(disp, min_disp=1.0, max_disp=256.0):
    """
    disparity가 너무 작거나(무한대) 음수면 신뢰 낮음.
    """
    valid = np.isfinite(disp) & (disp > min_disp) & (disp < max_disp)
    return valid


def fuse_depth(Z_stereo, stereo_valid_mask, risk_scdepth, z_floor=0.2, z_ceiling=8.0):
    """
    추천 융합:
    - 기본은 stereo Z (물리적 거리)
    - stereo가 깨지는 곳(valid False)은 scdepth 위험도를 대신 사용

    출력:
      fused_score: 0~1 위험도
      (가까울수록 1)
    """
    Z = Z_stereo.copy()

    # stereo Z가 말도 안되게 튀는 값 정리
    Z[~np.isfinite(Z)] = np.nan
    Z[(Z < z_floor) | (Z > z_ceiling)] = np.nan

    # stereo 기반 위험도 (가까울수록 1)
    # Z -> score: z_floor 근처 1, z_ceiling 근처 0
    stereo_score = np.zeros_like(Z, dtype=np.float32)
    ok = np.isfinite(Z) & stereo_valid_mask
    stereo_score[ok] = 1.0 - (Z[ok] - z_floor) / (z_ceiling - z_floor)
    stereo_score = np.clip(stereo_score, 0.0, 1.0)

    # scdepth 위험도는 0~1로 들어온다고 가정
    risk_scdepth = np.clip(risk_scdepth.astype(np.float32), 0.0, 1.0)

    # 융합: stereo 신뢰 높으면 stereo_score 우선, 아니면 scdepth
    fused = np.where(ok, stereo_score, risk_scdepth)

    return fused
