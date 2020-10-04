import cv2
import numpy as np

cap = cv2.VideoCapture(0)

_, frame = cap.read()

face_casc = cv2.CascadeClassifier('')