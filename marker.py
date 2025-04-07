import cv2
import numpy as np

aruco = cv2.aruco
dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
for marker_id in range(5):
    marker_img = aruco.generateImageMarker(dictionary, marker_id, 700)
    cv2.imwrite(f"marker_id{marker_id}.png", marker_img)

