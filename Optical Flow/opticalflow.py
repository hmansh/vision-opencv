import cv2
import numpy as np

#st_params 
st_params = dict(maxCorners = 30, qualityLevel = 0.2, minDistance = 2, blockSize =  7)

#lk parmas
lk_params = dict(winSize = (10, 10),
                    maxLevel = 4,
                    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03 ))

#captures the video cam feed
cap = cv2.VideoCapture(0)

#color
color = (0, 255, 0)

#read the capture and get the first frame
ret, first_frame = cap.read()

#conver the frame to grayscle
prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

#find the strongest corners in the first frame
prev = cv2.goodFeaturesToTrack(prev_gray, mask = None, **st_params)

#create and numpy array as mask with the same shape
mask = np.zeros_like(first_frame)

# #mouse fucntion
# def select_point(event, x, y, flags, params):
#     global point, point_selected, old_points
#     if event == cv2.EVENT_LBUTTONDOWN:
#         point = (x, y)
#         point_selected = True
#         old_points = np.array([[x, y]], dtype = np.float32)

# cv2.namedWindow("Frame")
# cv2.setMouseCallback("Frame", select_point)

# point_selected = False
# point = ()
# old_points = np.array([[]])

while (cap.isOpened()):
    ret , frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    next, status, error = cv2.calcOpticalFlowPyrLK(prev_gray, gray,  prev, None, **lk_params)

    good_old = prev[status == 1]

    good_new = next[status == 1]

    for i , (new, old) in enumerate(zip(good_new, good_old)):

        a, b = new.ravel()
        c, d = old.ravel()

        mask = cv2.line(mask, (a,b), (c, d), color , 2)

    #filled circle
        frame = cv2.circle(frame, (a, b),3, color, -1)


    output = cv2.add(frame, mask)

    prev_gray = gray.copy()

    prev = good_new.reshape(-1, 1, 2)

    cv2.imshow("Optical Flow", output)

    # if point_selected is True:
    #     cv2.circle(frame, 5, (0, 0, 255), 2)

        
    #     old_gray  = gray_frame.copy()
    #     old_points = new_points
    #     print(new_points)

    # cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cv2.destroyAllWindows(0)