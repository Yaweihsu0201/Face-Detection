import cv2
import numpy as np
from libsvm.svmutil import svm_load_model, svm_predict
import sys

# --- load model and normalization ---
model = svm_load_model("model5/face_svm.model")
mean = np.load("model5/mean.npy")
std = np.load("model5/std.npy")
std[std == 0] = 1.0

# --- read testing image ---
#image_path = "TestImagesForPrograms/8.jpg"
image_path = sys.argv[1]
output_path = sys.argv[2]
img_bgr = cv2.imread(image_path)

if img_bgr is None:
    raise FileNotFoundError(f"Cannot read image: {image_path}")

img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
height, width, _ = img_rgb.shape

# --- compute Y Cb Cr for all pixels at once ---
r = img_rgb[:, :, 0].astype(np.float32)
g = img_rgb[:, :, 1].astype(np.float32)
b = img_rgb[:, :, 2].astype(np.float32)

Y  = 0.299 * r + 0.587 * g + 0.114 * b
Cb = -0.168736 * r - 0.331264 * g + 0.5 * b
Cr = 0.5 * r - 0.418688 * g - 0.081312 * b

features = np.stack([Y, Cb, Cr], axis=-1).reshape(-1, 3)

# --- normalize ---
features = (features - mean) / std

# --- predict all pixels in one call ---
dummy_labels = [0] * len(features)
p_label, _, _ = svm_predict(dummy_labels, features.tolist(), model, '-q')

# --- reshape to binary image ---
binary = np.array(p_label, dtype=np.uint8).reshape(height, width) * 255

# --- PostProcessing ---
kernel = np.ones((5,5), np.uint8)

binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
binary = cv2.medianBlur(binary, 5)

num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

filtered = np.zeros_like(binary)

for i in range(1, num_labels):
    area = stats[i, cv2.CC_STAT_AREA]

    if area >= height*width/1000:
        filtered[labels == i] = 255

#k = int(width/100)
#kernel = np.ones((5,5), np.uint8)

# --- open ---
#eroded = cv2.erode(binary, kernel, iterations=k)
#binary = cv2.dilate(eroded, kernel, iterations=k)

# --- close ---
#dilated = cv2.dilate(binary, kernel, iterations=k)
#binary = cv2.erode(dilated, kernel, iterations=k)

# --- save the file ---
binary = filtered
cv2.imwrite(output_path, binary)
"""
cv2.imshow("Original", img_bgr)
cv2.imshow("Binary", binary)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""
print("image saved")