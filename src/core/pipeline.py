import cv2
import yaml

from src.detection.yolo_detector import YOLODetector
from src.tracking.byte_tracker import ByteTrackWrapper

from src.visualization.visualization import draw_tracks

class Pipeline:

    def __init__(self, config_path: str = "configs/pipeline.yaml") -> None:
        # ---- Load config ----
        with open(config_path, "r") as f:
            self.cfg = yaml.safe_load(f)

        # ---- Detector ----
        det_cfg = self.cfg["detector"]
        self.detector = YOLODetector(
            model_path=det_cfg["model_path"],
            conf=det_cfg["conf"],
            allowed_classes=det_cfg["allowed_classes"],
            img_size=det_cfg["img_size"]
        )

        # ---- Tracker ----
        trk_cfg = self.cfg["tracker"]
        self.tracker = ByteTrackWrapper(
            track_activation_threshold=trk_cfg["track_activation_threshold"],
            lost_track_buffer=trk_cfg["lost_track_buffer"],
            minimum_matching_threshold=trk_cfg["minimum_matching_threshold"],
            frame_rate=trk_cfg["frame_rate"]
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