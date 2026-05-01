import cv2

from src.detection.yolo_detector import YOLODetector
from src.tracking.byte_tracker import ByteTrackWrapper

from src.visualization.visualization import draw_detections, draw_tracks

class Pipeline:

    def __init__(self) -> None:
        self.detector = YOLODetector("models/yolo/yolo11m-seg.pt", conf=0.3)
        self.tracker = ByteTrackWrapper(
            track_activation_threshold=0.25,
            lost_track_buffer=30,
            minimum_matching_threshold=0.85,
            frame_rate=30
        )

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

                frame = cv2.resize(frame, None, fx=0.7, fy=0.7)  

                detections = self.detector.detect(frame)
                tracks = self.tracker.update(detections, frame)   

                frame = draw_tracks(frame, tracks)
                cv2.imshow("Frame", frame)

                if cv2.waitKey(1) == 27:
                    break

        finally:

            cap.release()
            cv2.destroyAllWindows()