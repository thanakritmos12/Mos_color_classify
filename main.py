import cv2
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from utils.color_detection import detect_color
from utils.visualization import draw_boxes

# ตั้งค่า Detectron2
def setup_detectron2():
    cfg = get_cfg()
    # ใช้โมเดล 'faster_rcnn_R_50_FPN_3x.yaml' ที่มีอยู่ใน Model Zoo
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.75  # ตั้งค่า Threshold
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    cfg.MODEL.DEVICE = "cpu"  # ใช้ CPU แทน GPU
    return DefaultPredictor(cfg)

# ตรวจจับคนและระบุสีเสื้อ
def detect_people_and_colors(predictor, frame):
    outputs = predictor(frame)
    instances = outputs["instances"].to("cpu")
    boxes = instances.pred_boxes.tensor.numpy()
    scores = instances.scores.numpy()
    classes = instances.pred_classes.numpy()

    detected_boxes = []
    labels = []

    for i, box in enumerate(boxes):
        if classes[i] == 0:  # Class 0 คือคนใน COCO dataset
            color = detect_color(frame, box)
            label = f"Person {i+1}: {color}"
            detected_boxes.append(box)
            labels.append(label)

    return detected_boxes, labels

# หลักการทำงาน
def main():
    predictor = setup_detectron2()
    cap = cv2.VideoCapture('data/videos/red2.mp4')

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ตรวจจับคนและระบุสีเสื้อ
        boxes, labels = detect_people_and_colors(predictor, frame)

        # วาดผลลัพธ์บนภาพ
        draw_boxes(frame, boxes, labels)

        # แสดงผล
        cv2.imshow('CCTV Footage', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
