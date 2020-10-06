import cv2
import numpy as np

img1 = cv2.imread("resources/ps0-1-a-1.png")
img2 = cv2.imread("resources/ps0-1-a-2.png")

print(img1.shape)
print(img2.shape)
while True:
    cv2.imshow("img1", img1)
    cv2.imshow("img2", img2)

    cv2.imshow("imgSwapped", cv2.cvtColor(img1, cv2.COLOR_RGB2BGR))

    cv2.imshow("imgRed", img1[:,:,0])
    cv2.imshow("imgGreen", img1[:,:,1])
    cv2.imshow("imgBlue", img1[:,:,2])

    img3 = img2[:,:,1].copy()
    x1, y1 = img3.shape
    w1 = int((x1-100)/2)
    z1 = int((y1-100)/2)
    img4 = img1[:, :, 1].copy()
    x2, y2 = img4.shape
    w2 = int((x2-100)/2)
    z2 = int((y2-100)/2)

    img3[ w1:w1+100, z1:z1+100] = img4[ w2:w2+100, z2:z2+100]
    cv2.imshow("Swapped", img3)
    mean = np.mean(img4)
    std = np.std(img4)

    img_green = cv2.add(cv2.multiply(cv2.divide(cv2.subtract(img4, mean), std), 10),mean)
    cv2.imshow("ImgGreen", img_green)

    t = np.float32([[1, 0, -2],[0, 1, 0]])
    transformed = cv2.warpAffine(img_green, t, (y2, x2))
    cv2.imshow("2pix",transformed)

    cv2.imshow("Difference", img_green - transformed)

    img_noise = img1.copy()
    noise = np.zeros_like(img_noise[:,:,1], dtype = np.uint8)
    cv2.randn(noise, 0, 30)
    img_noise[:,:,1] += noise
    cv2.imshow("Gaussian Noise 0-30", img_noise)

    if cv2.waitKey(1) == ord("q"):
        break;
    
print(np.mean(img4), np.std(img4), np.max(img4), np.min(img4))
cv2.destroyAllWindows()