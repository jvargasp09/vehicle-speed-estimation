import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple


def draw_detections(frame: np.ndarray, detections: List[Dict[str, Optional[float]]], color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
    """
    Draw raw detections (bounding boxes and labels) on a frame.

    Args:
        frame (np.ndarray): The input image frame.
        detections (List[Dict[str, Optional[float]]]): List of detections, each containing:
            - "bbox": The bounding box coordinates (x1, y1, x2, y2).
            - "confidence" (optional): The detection confidence.
            - "class" (optional): The class ID of the detected object.
        color (Tuple[int, int, int]): The color for drawing (default is green).

    Returns:
        np.ndarray: The frame with detections drawn.
    """
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        conf = det.get("confidence", 0.0)
        cls = det.get("class", -1)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        label = f"CLS:{cls} | {conf:.2f}"
        cv2.putText(frame, label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return frame


def draw_tracks(frame: np.ndarray, tracks: List[Dict[str, Optional[float]]], color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
    """
    Draw tracked objects with IDs on a frame.

    Args:
        frame (np.ndarray): The input image frame.
        tracks (List[Dict[str, Optional[float]]]): List of tracked objects, each containing:
            - "bbox": The bounding box coordinates (x1, y1, x2, y2).
            - "id": The object ID.
            - "confidence" (optional): The tracking confidence.
            - "class" (optional): The class ID of the tracked object.
        color (Tuple[int, int, int]): The color for drawing (default is green).

    Returns:
        np.ndarray: The frame with tracks drawn.
    """
    for obj in tracks:
        x1, y1, x2, y2 = obj["bbox"]
        obj_id = obj.get("id", -1)
        conf = obj.get("confidence", 0.0)
        cls = obj.get("class", -1)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        label = f"ID:{obj_id} | CLS:{cls} | {conf:.2f}"
        cv2.putText(frame, label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return frame
