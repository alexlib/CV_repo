import cv2 as cv
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
  
# The video feed is read in as
# a VideoCapture object
cap = cv.VideoCapture(r"Fire_Embers_20___25s___4k_res.mp4")
  
# ret = a boolean return value from
# getting the frame, first_frame = the
# first frame in the entire video sequence

#Added some annotations
first_frame: np.ndarray; frame: np.ndarray 
grey: np.ndarray; prev_gray: np.ndarray
flow: np.ndarray 

fb_params = dict(
    pyr_scale = .5, 
    levels = 2, 
    winsize = 5, 
    iterations = 3, 
    poly_n = 5, 
    poly_sigma = 1.2, 
    flags = 0
)

ret, first_frame = cap.read()
first_frame = cv.resize(first_frame, (960, 540))
# Converts frame to grayscale because we
# only need the luminance channel for
# detecting edges - less computationally 
# expensive
prev_gray = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)
  
# Creates an image filled with zero
# intensities with the same dimensions 
# as the frame
mask: np.ndarray = np.zeros_like(first_frame)
  
# Sets image saturation to maximum
mask[..., 1] = 255

fig, ax = plt.subplots()

while(cap.isOpened()):
      
    # ret = a boolean return value from getting
    # the frame, frame = the current frame being
    # projected in the video
    ret, frame = cap.read()
    frame = cv.resize(frame, (960, 540)) 
    # Opens a new window and displays the input
    # frame
    # cv.imshow("input", frame)
    ax.imshow(frame)
    

      
    # Converts each frame to grayscale - we previously 
    # only converted the first frame to grayscale
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
      
    # Calculates dense optical flow by Farneback method
    flow = cv.calcOpticalFlowFarneback(prev_gray, gray, flow=None, **fb_params)
      
    # Computes the magnitude and angle of the 2D vectors
    magnitude, angle = cv.cartToPolar(flow[..., 0], flow[..., 1])
      
    # Sets image hue according to the optical flow 
    # direction
    mask[..., 0] = angle * 180 / np.pi / 2
      
    # Sets image value according to the optical flow
    # magnitude (normalized)
    mask[..., 2] = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX)
      
    # Converts HSV to RGB (BGR) color representation
    rgb: np.ndarray = cv.cvtColor(mask, cv.COLOR_HSV2BGR)

    # Opens a new window and displays the output frame
    # cv.imshow("dense optical flow", rgb)
    ax.imshow(rgb)
      
    # Updates previous frame
    prev_gray = gray
      
    # Frames are read by intervals of 1 millisecond. The
    # programs breaks out of the while loop when the
    # user presses the 'q' key
    # if cv.waitKey(1) & 0xFF == ord('q'):
    #     break
  
# The following frees up resources and
# closes all windows
cap.release()
cv.destroyAllWindows()