from typing import List, Optional
import numpy as np
import supervision as sv

from src.tracking.base_tracker import BaseTracker


class ByteTrackWrapper(BaseTracker):
    """
    Wrapper for the ByteTrack implementation using the 'supervision' library.
    Adapts ByteTrack to conform to the BaseTracker interface.
    """

    def __init__(
        self,
        track_activation_threshold: float = 0.25,
        lost_track_buffer: int = 30,
        minimum_matching_threshold: float = 0.85,
        frame_rate: int = 30
    ):
        """
        Initializes the ByteTrackWrapper with the given parameters.
        
        Args:
            track_activation_threshold (float): Threshold for activating a new track.
            lost_track_buffer (int): Number of frames to keep a track in case of missing detections.
            minimum_matching_threshold (float): Threshold for the minimum confidence to match a detection with a track.
            frame_rate (int): Frame rate of the video for the tracker.
        """
        # Create an instance of the ByteTrack tracker using the provided configuration.
        self.tracker = sv.ByteTrack(
            track_activation_threshold=track_activation_threshold,
            lost_track_buffer=lost_track_buffer,
            minimum_matching_threshold=minimum_matching_threshold,
            frame_rate=frame_rate
        )

    def update(self, detections: List[dict], frame: Optional[np.ndarray] = None):
        """
        Converts detections into the format expected by supervision, 
        runs ByteTrack, and returns results in a standard format.

        Args:
            detections (List[dict]): List of detections, each with bbox, confidence and class.
            frame (Optional[np.ndarray]): Optional frame for visualizations (not used here).

        Returns:
            List[dict]: A list of tracks, each represented as a dictionary with 'id', 'bbox', 'confidence' and 'class'.
        """

        # If there are no detections, return an empty list
        if detections is None or len(detections) == 0:
            return []

        # ---- Convert detections to numpy arrays ----
        boxes = np.array([d["bbox"] for d in detections], dtype=np.float32)  # Bounding boxes
        confidences = np.array([d["confidence"] for d in detections], dtype=np.float32)  # Confidence scores
        class_ids = np.array([d["class"] for d in detections], dtype=int)  # Class IDs



        # ---- Create Supervision Detections object ----
        sv_detections = sv.Detections(
            xyxy=boxes,
            confidence=confidences,
            class_id=class_ids
        )

        # ---- Track the detections ----
        tracked = self.tracker.update_with_detections(sv_detections)  # Run ByteTrack on detections

        # ---- Process tracking results ----
        results = []
        for i, (xyxy, track_id, conf, cls) in enumerate(zip(
            tracked.xyxy,       # Bounding boxes of tracked objects
            tracked.tracker_id, # Track IDs
            tracked.confidence, # Confidence scores of the tracks
            tracked.class_id    # Class IDs of the tracked objects
        )):
            results.append({
                "id":         int(track_id),  # Track ID
                "bbox":       tuple(map(int, xyxy)),  # Bounding box as tuple of integers
                "confidence": float(conf) if conf is not None else 0.0,  # Confidence score (default to 0 if None)
                "class":      int(cls) if cls is not None else -1,  # Class ID (default to -1 if None)
            })

        return results

    def reset(self):
        """
        Resets the tracker by creating a new instance of the tracker.
        """
        self.tracker.reset()  # Reset the ByteTrack tracker instance