import cv2


def open_scdepth_shm(socket_path="/tmp/scdepth.sock"):
    """
    gst-launch에서 shmsink로 뿌린 RGB 프레임을 OpenCV로 받는다.
    """
    pipe = (
        f"shmsrc socket-path={socket_path} do-timestamp=true is-live=true ! "
        f"video/x-raw,format=RGB,width=320,height=256,framerate=30/1 ! "
        f"videoconvert ! video/x-raw,format=BGR ! appsink drop=true max-buffers=1 sync=false"
    )
    cap = cv2.VideoCapture(pipe, cv2.CAP_GSTREAMER)
    return cap


def scdepth_risk_from_colormap(bgr):
    """
    scdepth 결과는 '컬러맵'이라 절대깊이는 못 뽑음.
    대신 근접 위험(상대)을 뽑자:
      - 보통 가까운 영역이 강하게 표시되는 색이 있음(모델/overlay 설정에 따라 다름)
    가장 단순/강건하게:
      - HSV의 V(밝기) 기반으로 "강한 영역"을 위험으로 취급
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    v = hsv[..., 2].astype("float32") / 255.0  # 0~1
    # 위험도: 밝을수록 위험이라고 가정(너 overlay가 그렇게 보이는 경우가 많음)
    # 만약 반대면 (1-v)로 뒤집으면 됨.
    risk = v
    return risk
