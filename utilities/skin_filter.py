import cv2
import numpy as np
from libsvm.svmutil import svm_load_model, svm_predict
import os
import matplotlib.pyplot as plt
import sys

# --- load model and normalization ---
model = svm_load_model("model5/face_svm.model")
mean = np.load("model5/mean.npy")
std = np.load("model5/std.npy")
std[std == 0] = 1.0

gamma_value = 1.2
clahe_clip = 2.0
clahe_grid = (8, 8)

def gamma_correction(img_bgr, gamma=1.2):
    inv_gamma = 1.0 / gamma
    table = np.array([
        ((i / 255.0) ** inv_gamma) * 255
        for i in np.arange(256)
    ]).astype(np.uint8)
    return cv2.LUT(img_bgr, table)


def apply_clahe_on_y(img_bgr, clip_limit=2.0, tile_grid_size=(8, 8)):
    # OpenCV 的 YCrCb 順序是 [Y, Cr, Cb]
    img_ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    Y = img_ycrcb[:, :, 0]
    Cr = img_ycrcb[:, :, 1]
    Cb = img_ycrcb[:, :, 2]

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    Y_eq = clahe.apply(Y)

    out_ycrcb = np.stack([Y_eq, Cr, Cb], axis=-1)
    out_bgr = cv2.cvtColor(out_ycrcb, cv2.COLOR_YCrCb2BGR)
    return out_bgr


def preprocess_image(img_bgr):
    img_bgr = gamma_correction(img_bgr, gamma=gamma_value)
    img_bgr = apply_clahe_on_y(
        img_bgr,
        clip_limit=clahe_clip,
        tile_grid_size=clahe_grid
    )
    return img_bgr

def predict_in_batches(features, model, batch_size=50000):
    n = features.shape[0]
    all_pred = []

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = features[start:end]
        dummy_labels = [0] * len(batch)

        p_label, _, _ = svm_predict(dummy_labels, batch.tolist(), model, '-q')
        all_pred.extend(p_label)

        print(f"processed {end}/{n}", end="\r")

    return np.array(all_pred)

def skinfilter(img_rgb):
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
    p_label = predict_in_batches(features, model, batch_size=50000)

    # --- reshape to binary image ---
    binary = np.array(p_label, dtype=np.uint8).reshape(height, width) * 255

    # --- PostProcessing ---
    kernel = np.ones((5,5), np.uint8)

    #binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    #binary = cv2.medianBlur(binary, 5)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

    filtered = np.zeros_like(binary)

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]

        if area >= height*width/1000:
            filtered[labels == i] = 255
    return filtered

if __name__ == "__main__":
    image_path = sys.argv[1]
    output_path = sys.argv[2]
    img_bgr = cv2.imread(image_path)
    img = preprocess_image(img_bgr)
    #img = img_bgr
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    filtered = skinfilter(img_rgb)
    cv2.imwrite(output_path, filtered)
    print("image saved")