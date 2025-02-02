import cv2

def draw_boxes(image, boxes, labels):
    """
    วาด bounding box และ label บนภาพ
    :param image: ภาพต้นฉบับ
    :param boxes: รายการ bounding boxes [[x1, y1, x2, y2], ...]
    :param labels: รายการ labels สำหรับแต่ละ box
    """
    for box, label in zip(boxes, labels):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)