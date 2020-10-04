import cv2
import numpy as np
import matplotlib.pyplot as plt

original_image = cv2.imread("")
hsv_original = cv2.cvtColor(original_image, cv2,COLOR_BGR2HSV)

roi = cv2.imread("")
hsv_roi = cv2.cvtColor(roi, cv2,COLOR_BGR2HSV)

hue, saturation, value = cv2.split(hsv_roi)

#hsv 0- 180, saturation - 0 - 256
roi_hist = cv2.calcHist([hsv_roi], [0, 1], None, [180, 256], [0, 180, 0 , 256])
mask = cv2.calcBackProject([hsv_original], [0, 1], roi_hist, [0, 180, 0 , 256], 1)

#filtering noise
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
#[1, 2,3 ,4,4]
cv2.filter2D(mask, -1, kernel)

#better threshold filtering
mask = cv2.threshold(mask, 30 , 155, cv2.THRESH_BINARY)

#because image is of 3 chanels
mask - cv2.merge((mask, mask, mask))
result = cv2.bitwise_and(original_image, mask)


# cv2.imshow("Original Image", original_image)
# cv2.imshow("Roi", roi)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

plt.imshow(roi_hist)
plt.show()