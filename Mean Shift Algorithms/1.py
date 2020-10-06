import cv2
import numpy as np 

video = cv2.VideoCapture(0)
_, frame = video.read()
(x, y, w, h) = cv2.selectROI(frame, False)
# print(roi)
# img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# gray = img.copy()
# print(img.shape)
# for i in range(0, img.shape[0]):
    # for j in range(0, img.shape[1]):y:y+h,
        # img[i][j]= int(img[i][j]/20)*20
roi = frame[x:x+w,y:y+h]
cv2.imshow("Mask", roi)
hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
roi_hist = cv2.calcHist([hsv_roi], [0], None, [255], [0, 255])
roi_hist = cv2.normalize(roi_hist, roi_hist, 0, 300, cv2.NORM_MINMAX)

term_criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

while True:
    # cv2.imshow("Image", frame)
    # cv2.imshow("Gray Scale", gray)
    # cv2.imshow("Image Edit", img)
    _, frame = video.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.calcBackProject([hsv], [0], roi_hist, [0, 255], 1)

    _, track_window = cv2.meanShift(mask, (x, y, w, h), term_criteria)
    x, y, w, h = track_window
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Mask", mask)
    cv2.imshow("Frame", frame)

    if cv2.waitKey(0) == ord('q'):
        break
    
video.release()
cv2.destroyAllWindows()