import cv2
import matplotlib.pyplot as plt
import numpy as np

mean = np.load("model3/mean.npy")
std = np.load("model3/std.npy")
std[std == 0] = 1.0

def get_ycbcr_channels_norm(path):
    img_bgr = cv2.imread(path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_ycrcb = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YCrCb)

    Y  = img_ycrcb[:, :, 0].astype(np.float32)
    Cr = img_ycrcb[:, :, 1].astype(np.float32)
    Cb = img_ycrcb[:, :, 2].astype(np.float32)

    features = np.stack([Y, Cb, Cr], axis=-1).reshape(-1, 3)

    features_norm = (features - mean) / std

    Y_norm  = features_norm[:, 0]
    Cb_norm = features_norm[:, 1]
    Cr_norm = features_norm[:, 2]

    return Y_norm, Cb_norm, Cr_norm


Y1, Cb1, Cr1 = get_ycbcr_channels_norm("TestImagesForPrograms/lab2.jpg")
Y2, Cb2, Cr2 = get_ycbcr_channels_norm("TestImagesForPrograms/2.jpg")

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.hist(Y1, bins=50, alpha=0.5, label="good")
plt.hist(Y2, bins=50, alpha=0.5, label="bad")
plt.title("Y (normalized)")
plt.legend()

plt.subplot(1, 3, 2)
plt.hist(Cb1, bins=50, alpha=0.5, label="good")
plt.hist(Cb2, bins=50, alpha=0.5, label="bad")
plt.title("Cb (normalized)")
plt.legend()

plt.subplot(1, 3, 3)
plt.hist(Cr1, bins=50, alpha=0.5, label="good")
plt.hist(Cr2, bins=50, alpha=0.5, label="bad")
plt.title("Cr (normalized)")
plt.legend()

plt.tight_layout()
plt.show()