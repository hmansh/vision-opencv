import numpy as np
import cv2

def detectCircles(img, val = 21, minR = 0, maxR = 0):
    blur = cv2.GaussianBlur(img, (val, val), 0)
    all_circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, 1, 20, param1 = 50,
                                        param2 = 30, minRadius = minR, maxRadius = maxR)
    return np.uint16(np.round(all_circles))
    
def findEgdes(img, threshold = 0.33, val = 5):
    # img = cv2.GaussianBlur(img, (val, val), 0)
    # med = np.median(img)
    # lower = int(max(0, (1.0 - threshold)*med))
    # upper = int(min(0, (1.0 + threshold)*med))
    return cv2.Canny(img, 10, 50)

def detectLines(edges):
    # edges = findEgdes(img)
    lines = cv2.HoughLines(edges, 1, np.pi/180, 200 )
    results = []
    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int( x0 + 1000 * (-b))
        y1 = int( y0 + 1000 * a )
        x2 = int( x0 - 1000 * (-b))
        y2 = int( y0 - 1000 * (a))
        results.append([(x1, y1), (x2, y2)])
    return results


img = cv2.imread("/home/himanshu/Object Detection/Res/resources/ps1-input1.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print("gray.shape")

edges = findEgdes(gray)
lines = detectLines(edges)
circles = detectCircles(gray)

print(len(edges))
# print(len(lines))
# print(lines[1][0])
# print(len(circles))
# print(circles[0])
l = img.copy()
while True:
    cv2.imshow("Image", img)
    cv2.imshow("Gray", gray)
    # for line in lines:
        # l = cv2.line(l, line[0], line[1], (0, 0, 255), 3)

    for circle in circles[0]:
        print(circle)
        cv2.circle(l, (circle[0], circle[1]), circle[2], (0, 255, 0), 2)
    
    cv2.imshow("Lines", l)

    if cv2.waitKey(0) == ord('q'):
        break

cv2.destroyAllWindows()

