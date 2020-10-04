import numpy as np
import cv2
from random import randint

tracker_types = ['BOOSTING',
                'MIL',
                'KCF',
                'TLD',
                'MEDIANFLOW',
                'GOTURN',
                'MOSSE',
                'CSRT']

def traker_name(tracker):
    if tracker == tracker_types[0]:
        return cv2.TrackerBoosting_create()

    if tracker == tracker_types[1]:
        return cv2.TrackerMIL_create()
    
    if tracker == tracker_types[2]:
        return cv2.TrackerKCF_create()
    
    if tracker == tracker_types[3]:
        return cv2.TrackerTLD_create()
    
    if tracker == tracker_types[4]:
        return cv2.TrackerMedianFlow_create()
    
    if tracker == tracker_types[5]:
        return cv2.TrackerGOTURN_create()
    
    if tracker == tracker_types[6]:
        return cv2.TrackerMOSSE_create()

    if tracker == tracker_types[7]:
        return cv2.TrackerCSRT_create()
    
    else :
        raise NotImplementedError


if __name__ == "__main__":
    print("Default Tracker MOSSE\n")

    tracker = 'MOOSE'

    cap = cv2.VideoCature(0)
    
    _, frame = cap.read()

    rects = []
    colors = []

    while True:

        rect_box = cv2.selectROI('MultiTracker', frame)
        rects.append(rect_box)
        colors.append((randint(64, 255), randint(64, 255), randint(64, 255)))
        print("Enter q to Stop Tracking")
        print("Any other to continue selecting boxes.")

        if cv2.waitKey(0) == ord('q'):  
            break
    
    print(f"Selected Boxes {rects}")

    multitracker = cv2.MultiTracker_create()

    for rect_box in rects:
        multitracker.add(tracker_name(tracker_types), frame, rect_box)

    while cap.IsOpened():
    
        success, frame = cap.read()
        if not success:
            break

        success, boxrs = multitracker.update(frame)

        for i, window in enumerate(boxes):
            pts1 = (int(newbox[0]),
            int(newbox[1]))

cv2.release()
cv2.destroyAllWindows()
