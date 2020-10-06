import cv2
import numpy as np

st_params = dict(maxCorners = 30, qualityLevel = 0.2,
                    minDistance = 2, blockSize = 7 )

# lk_params = dict(winSize = (10, 10), 
#                     maxLevel = 4,
#                     criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))


cap = cv2.VideoCapture(0)

color = (0, 255, 0)

_, first_frame = cap.read()

prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

prev = cv2.goodFeaturesToTrack(prev_gray, mask = None, **st_params)

mask = np.zeros_like(first_frame)

mask[..., 1] = 255

while(cap.isOpened()):
    _, frame = cap.read()
    cv2.imshow("Input", frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    magn, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    #set image hue depending on the optical flow direction
    mask[..., 0] = angle*180/np.pi/2

    #normalize the magnitude
    mask[..., 2] = cv2.normalize(magn, None, 0, 255, cv2.NORM_MINMAX)

    #conver HSV to RGB
    rgb = cv2.cvtColor(mask, cv2.COLOR_HSV2RGB)

    #open window
    cv2.imshow("Optical Flow", rgb)

    prev_gray = gray

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cv2.destroyAllWindows(0)

