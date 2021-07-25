import cv2 as cv
import numpy as np
from typing import Union

class Points_tracker:
    # Reading capture from file or cam
    # Create named window to use different commands on the same window
    # Change resolution if needed
    # Add type annotations
    def __init__(self, num_points: int):
        self.old_points: list
        self.points: list = []
        self.sourse = None
        self.points_selected: bool = False
        self.num_points: int = num_points

        self.lk_params = dict(
            winSize  = (15, 15), # Size of detection window
            maxLevel = 2, # Pyramid level
            criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03)
        )

    def set_sourse(self, name: Union[str, int]):
        self.sourse = name

    # Callback function to get start points
    def _select_points(self, event, x: int, y: int, flags, params) -> None:
        if event == cv.EVENT_LBUTTONDOWN:
            if len(self.points) < self.num_points:
                self.points.append([x, y])
            else:
                self.points_selected = True
                self.old_points = np.array(self.points, dtype=np.float32)
                cv.setMouseCallback("Frame", lambda *args : None)


    def _draw_circle(self, point: list[int]):
        cv.circle(self._frame, tuple(point), radius=5, color=(255, 0, 0), thickness=-1) 


    def track_points(self):
        if self.sourse:
            cap = cv.VideoCapture(self.sourse)
        else:
            return
        
        cv.namedWindow("Frame")
        cv.setMouseCallback("Frame", self._select_points)
        ret, old_frame = cap.read()
        old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
        
        while True:

            if not ret or cv.waitKey(1) == 27:
                    break
            
            if self.points_selected is True:

                ret, self._frame = cap.read()
                
                # Exit on "esc" button
                gray = cv.cvtColor(self._frame, cv.COLOR_BGR2GRAY)
                
                new_points, status, error = cv.calcOpticalFlowPyrLK(old_gray, gray, self.old_points, 
                                                                    None, **self.lk_params)
                old_gray = gray.copy()
                self.old_points = new_points
                list(map(self._draw_circle, new_points.astype(int).tolist()))

                cv.imshow("Frame", self._frame)
            else:
                cv.imshow("Frame", old_frame)

        cap.release()
        cv.destroyWindow("Frame")

if __name__ == "__main__":
    pt = Points_tracker(num_points=2)
    pt.set_sourse("airport_airplanes_landing_strip_1008.mp4")
    pt.track_points()

