import cv2
import numpy as np

def detect_color(image, bbox):
    """
    ตรวจจับสีเสื้อจาก bounding box
    :param image: ภาพต้นฉบับ
    :param bbox: bounding box ของคนที่ตรวจจับได้ [x1, y1, x2, y2]
    :return: สีที่ตรวจจับได้ (str)
    """
    x1, y1, x2, y2 = map(int, bbox)
    cropped_image = image[y1:y2, x1:x2]
    hsv_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2HSV)

    # ตัวอย่างการตรวจจับสีแดง
    lower_red = np.array([0, 50, 50])
    upper_red = np.array([10, 255, 255])
    mask = cv2.inRange(hsv_image, lower_red, upper_red)

    if np.any(mask):
        return "Red"
    return "Unknown"