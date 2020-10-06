import numpy as np 
import cv2
from time import time

def similarity(img1, img2, std = 10):
    if np.subtract(img1.shape, img2.shape).any():
        return 0
    else:
        mse = np.sum(np.subtract(img1, img2, dtype=np.float32) ** 2)
        mse /= float(img1.shape[0] * img1.shape[1])
        return np.exp(-mse / 2 / std**2)

class ParticleFilter():

    def __init__(self, roi, search_space, num_particles, state_dims):
        self.roi = roi
        self.search_space = search_space
        self.num_particles = num_particles
        self.state_dims = state_dims
        self.particles = np.array([np.random.uniform(0, self.search_space[i], self.num_particles) 
                                                                for i in range(self.state_dims)]).T 
        self.weights = np.ones(len(self.particles))/len(self.particles)
        self.idxs = np.arange(self.num_particles)
        self.estimateSample()
        
    def update(self, frame):
        self.scatterParticles()
        self.getPatches(frame)
        self.resample()
        self.estimateSample()

    def scatterParticles(self):
        self.particles += np.random.normal(0, 15, self.particles.shape)

    def getPatches(self, frame):
        maxh, maxw = self.roi.shape[:2]
        minx = (self.particles[:, 0] - maxw/2).astype(np.int)
        miny = (self.particles[:, 1] - maxh/2).astype(np.int)
        candidates = [frame[miny[i]:miny[i] + maxh, minx[i]:minx[i] + maxw] for i in range(self.num_particles)]
        self.weights = np.array([similarity(cand, self.roi) for cand in candidates])
        self.weights /= np.sum(self.weights)

    def resample(self):
        j = np.random.choice(self.idxs, self.num_particles, True, p = self.weights.T)
        self.particles = np.array(self.particles[j])

    def estimateSample(self):
        idx = np.random.choice(self.idxs, 1, p = self.weights)
        self.state = self.particles[idx][0].astype(np.float16)

    def visualiseFilter(self, img):
        for p in self.particles:
            cv2.circle(img, tuple(p.astype(int)), 2, (0, 0, 255), -1)


capture = cv2.VideoCapture("videoplayback (1).mp4")
_, frame = capture.read()
gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
x, y, w, h = cv2.selectROI(gray, False)
roi = gray[y:y+h, x:x+w]

search_space = np.array(gray.shape)
num_particles = 100
state_dims = 2

tracker = ParticleFilter(roi, search_space, num_particles, state_dims)

while capture.isOpened():
    start_time = time()
    ret , frame = capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    tracker.update(gray)
    tracker.visualiseFilter(frame)

    delay = int(25 - (time() - start_time))
    if cv2.waitKey(delay) == ord('q'):
        break

    cv2.imshow("Video Playback", frame)

capture.release()
cv2.destroyAllWindows()