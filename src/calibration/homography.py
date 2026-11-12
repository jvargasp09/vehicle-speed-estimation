import numpy as np
import cv2

class Homography:

    def __init__(self, image_points: np.ndarray, world_points: np.ndarray):
        """
        Initialize homography matrix using selected calibration points.
        """        

        self.H, status = cv2.findHomography(image_points, world_points, 0)
        inliers = int(status.sum())
        print(f"Homography computed — inliers: {inliers}/{len(image_points)}")
        print("H:\n", self.H)    

    
    def project_point(self, point: tuple[float, float]) -> tuple[float, float]:
        """
        Project point to world coordinates.       
        """
    
        px, py = point
        world = self.H @ np.array([px, py, 1.0])
        world /= world[2]
        return float(world[0]), float(world[1])