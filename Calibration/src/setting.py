import cv2
import numpy as np
from pathlib import Path

output_path = Path(__file__).parent.parent / "output"
output_path.mkdir(parents=True, exist_ok=True)

# --- 参数设置 ---
def charuco_board():
    charuco_board = cv2.aruco.CharucoBoard(
        size=(4, 6),
        squareLength=0.017,
        markerLength=0.0125,
        dictionary=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),
    )
    return charuco_board


def charuco_parameters():
    charuco_params = cv2.aruco.CharucoParameters()
    charuco_params.cameraMatrix = np.array([
        [914.18157959, 0., 643.12854004],
        [0., 912.53094482, 363.66064453],
        [0., 0., 1.]
    ])
    charuco_params.distCoeffs = np.array([
        0., 0., 0., 0., 0.
    ])
    charuco_params.minMarkers = 2
    charuco_params.tryRefineMarkers = True
    return charuco_params


def detector_parameters():
    detector_params = cv2.aruco.DetectorParameters()
    detector_params.minDistanceToBorder = 3
    detector_params.useAruco3Detection = True
    detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector_params.minSideLengthCanonicalImg = 16
    detector_params.errorCorrectionRate = 0.8
    return detector_params


def refine_parameters():
    refine_params = cv2.aruco.RefineParameters()
    refine_params.minRepDistance = 10
    refine_params.errorCorrectionRate = 3
    refine_params.checkAllOrders = True
    return refine_params

