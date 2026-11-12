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

    def run(self, input_path: str, output_path: str | None = None) -> None:
        cap = cv2.VideoCapture(input_path)

        if not cap.isOpened():
            print(f"Error: Cannot open video {input_path}")
            return

        # ---- Output video ----
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            # Some containers/streams do not report FPS
            fps = 30.0
            print("Warning: FPS not available, falling back to 30")

        writer = None

        try:

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.resize(frame, None, fx=0.7, fy=0.7)

                # Create the writer once the resized frame size is known
                if output_path is not None and writer is None:
                    height, width = frame.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                    if not writer.isOpened():
                        raise RuntimeError(f"Cannot open video writer for {output_path}")

                detections = self.detector.detect(frame)
                tracks = self.tracker.update(detections, frame)

                frame = draw_tracks(frame, tracks)

                if writer is not None:
                    writer.write(frame)

                cv2.imshow("Frame", frame)

                if cv2.waitKey(1) == 27:
                    break

        finally:

            cap.release()
            if writer is not None:
                writer.release()
            cv2.destroyAllWindows()