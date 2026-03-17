import cv2
import os

DICT = cv2.aruco.DICT_6X6_250
dictionary = cv2.aruco.getPredefinedDictionary(DICT)

save_dir = "markers_6x6_250"
os.makedirs(save_dir, exist_ok=True)

for marker_id in range(0, 30):  # 这里举例生成 0~29 共 30 张
    img = cv2.aruco.generateImageMarker(dictionary, marker_id, 600, borderBits=1)
    cv2.imwrite(os.path.join(save_dir, f"id_{marker_id:03d}.png"), img)

print("Done:", save_dir)