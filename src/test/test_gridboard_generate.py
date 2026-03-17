import cv2

# 选择字典
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)

# 参数：markersX, markersY, markerLength(米), markerSeparation(米)
# 如果只为了打印，单位随意，但两者比例要正确；常用“像素”为单位也可以
markersX = 2
markersY = 3
markerLength = 0.045       # 例如 3 cm => 0.03 m （仅当做比例使用）
markerSeparation = 0.006  # 0.6 cm

board = cv2.aruco.GridBoard((markersX, markersY), markerLength, markerSeparation, dictionary)

# 生成一张大图（像素维度），用于打印
out_w, out_h = 2480, 3508  # 接近 A4 @ 300DPI（只是像素大小，不含真正 DPI 元数据）
img = board.generateImage((out_w, out_h), marginSize=20, borderBits=1)

cv2.imwrite(f"gridboard_{markersX}x{markersY}_5x5_100.png", img)