import cv2

from src.detection.yolo_detector import YOLODetector
from src.visualization.visualization import draw_detections, draw_tracks

class Pipeline:

    def __init__(self) -> None:
        self.detector = YOLODetector("models/yolo/yolo11m-seg.pt", conf=0.3)

    def run(self, input_path: str) -> None:
        cap = cv2.VideoCapture(input_path)

        if not cap.isOpened():
            print(f"Error: Cannot open video {input_path}")
            return
        
        try:

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                detections = self.detector.detect(frame)
                frame = draw_detections(frame, detections)
                cv2.imshow("Frame", frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:

            cap.release()
            cv2.destroyAllWindows()