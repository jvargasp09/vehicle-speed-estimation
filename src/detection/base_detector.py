from abc import ABC, abstractmethod


class BaseDetector(ABC):
    """
    Abstract base class for object detectors.

    This allows different detection implementations
    (YOLO, DETR, etc.) to be swapped without changing the pipeline.
    """

    @abstractmethod
    def detect(self, frame):
        """
        Run object detection on a single frame.

        Args:
            frame (np.ndarray): Input image (BGR or RGB).

        Returns:
            List[dict]: List of detections in standardized format:
                {
                    "bbox": (x1, y1, x2, y2),
                    "confidence": float,
                    "class": int
                }
        """
        pass