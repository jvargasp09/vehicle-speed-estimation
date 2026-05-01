from abc import ABC, abstractmethod


class BaseTracker(ABC):
    """
    Abstract base class for multi-object trackers.

    A tracker takes detections and returns objects with persistent IDs.
    """

    @abstractmethod
    def update(self, detections, frame=None):
        """
        Update tracker state with new detections.

        Args:
            detections (List[dict]):
                [
                    {
                        "bbox": (x1, y1, x2, y2),
                        "confidence": float,
                        "class": int
                    }
                ]
            frame (np.ndarray, optional): current frame (some trackers need it)

        Returns:
            List[dict]:
                [
                    {
                        "id": int,
                        "bbox": (x1, y1, x2, y2),
                        "confidence": float,
                        "class": int
                    }
                ]
        """
        pass

    @abstractmethod
    def reset(self):
        """
        Reset tracker state.
        """
        pass