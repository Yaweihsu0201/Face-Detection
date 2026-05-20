#utilities/mouthmap.py

import cv2
from matplotlib import pyplot as plt
from scipy.signal import convolve2d
import numpy as np

def tl(x,y):
    v = x+y-127.5**2 
    return np.maximum(v, 0)

def mouthmap(img_rgb, all_points):
    img_ycrcb = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YCrCb)
    Cr = img_ycrcb[:, :, 1]+127.5
    Cb = img_ycrcb[:, :, 2]+127.5
    xs = all_points[:, 0]
    ys = all_points[:, 1]

    eta = 0.95 * np.mean(Cr[ys, xs]**2) / np.mean(Cr[ys, xs] / (Cb[ys, xs] + 1e-8))
    eps = 1e-8

    A = Cr**2
    B = (Cr**2 - eta * (Cr / (Cb + eps)))**2

    MouthMap = tl(A, B)

    return MouthMap
