import cv2
import numpy as np

img = cv2.imread("image.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
corners = cv2.goodFeaturesToTrack(gray, 50, 0.01, 10)
print(np.int16(corners))

for i in corners:
    x, y = i.ravel()
    cv2.circle(img, (x, y), 1, 255, 5)

cv2.imshow("IMAGE", img)
cv2.waitKey(0)
cv2.destroyAllWindows()