import torch
import numpy as np
from ultralytics import YOLO
from src.detection.base_detector import BaseDetector


class YOLODetector(BaseDetector):
    """
    YOLO-based object detector that extends the BaseDetector class.
    It loads the YOLO model and performs object detection on input frames.
    """

    def __init__(self, model_path="models/yolo/yolo11n.pt", conf=0.5, allowed_classes=[2, 3, 5, 7], img_size=640):
        """
        Initializes the YOLO detector with the given model and confidence threshold.
        Args:
            model_path (str): Path to the YOLO model weights.
            conf (float): Confidence threshold for detection.
            allowed_classes (list): List of classes to detect. If None, all classes are detected.
            img_size (int): Image size for inference.
        """
        # Load the YOLO model from the specified path
        self.model = YOLO(model_path)

        # Set device to GPU if available, otherwise fallback to CPU
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(device)  # Move the model to the selected device

        self.device = device  # Store the device information
        self.half = device == "cuda"  # Use half precision on CUDA (GPU) if available

        self.conf = conf  # Set the confidence threshold for filtering detections
        self.allowed_classes = allowed_classes # Set the list of allowed classes for detection (e.g., vehicles)

        self.imgsz = img_size  # Image size for inference (can be adjusted based on model requirements)


    def detect(self, frame):
        """
        Perform detection on the provided frame.
        Args:
            frame (ndarray): Input image/frame for detection.
        Returns:
            list: List of detected objects with bounding boxes, confidence and class.
        """
        # Run the YOLO model on the frame
        results = self.model(
            frame,
            classes=self.allowed_classes,  # Only detect specified classes
            verbose=False,  # Disable verbose logging
            half=self.half,  # Use half precision if running on CUDA
            imgsz=self.imgsz  # Image size for inference (adjust based on model)
        )[0]

        detections = []  # List to store detection results

        # If no bounding boxes were detected, return an empty list
        if results.boxes is None:
            return []

        # Process each detected object
        for i, box in enumerate(results.boxes):
            cls = int(box.cls[0])  # Get the class index of the detected object
            conf = float(box.conf[0])  # Get the confidence score of the detection

            # Skip detections that are not in the allowed classes or below the confidence threshold
            if cls not in self.allowed_classes or conf < self.conf:
                continue

            # Get the bounding box coordinates for the detected object
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Append the detection to the list of detections
            detections.append({
                "bbox": (x1, y1, x2, y2),
                "confidence": conf,
                "class": cls
            })

        return detections  # Return the list of detections