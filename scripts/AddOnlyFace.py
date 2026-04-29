import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

# ===== 設定 =====
adding_list = ["TestImagesForPrograms/44.jpg","TestImagesForPrograms/06.jpg"]
#adding_list = ["TestImagesForPrograms/19.jpg"]
output_file = "model5/training_raw.txt"

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


def collect_points(n, title, img_bgr):
    img_rgb_show = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(8, 6))
    plt.imshow(img_rgb_show)
    plt.title(title)
    plt.axis("on")

    pts = plt.ginput(n, timeout=0)
    plt.close()

    return pts


def extract_features(points, img_rgb, label):
    height, width, _ = img_rgb.shape
    records = []

    for x, y in points:
        px = int(round(x))
        py = int(round(y))

        px = max(0, min(px, width - 1))
        py = max(0, min(py, height - 1))

        r, g, b = img_rgb[py, px].astype(np.float32)

        Y  = 0.299 * r + 0.587 * g + 0.114 * b
        Cb = -0.168736 * r - 0.331264 * g + 0.5 * b
        Cr = 0.5 * r - 0.418688 * g - 0.081312 * b

        records.append([label, Y, Cb, Cr])

    return records


data = []

for element in adding_list:
    sample_bgr = cv2.imread(element)
    if sample_bgr is None:
        print(f"skip: cannot read {element}")
        continue

    # ===== 前處理 =====
    processed_bgr = preprocess_image(sample_bgr)
    processed_rgb = cv2.cvtColor(processed_bgr, cv2.COLOR_BGR2RGB)

    # ===== 手動取樣 =====
    face_points = collect_points(5, f"Click 5 FACE points: {element}", processed_bgr)
    #nonface_points = collect_points(10, f"Click 10 NON-FACE points: {element}", processed_bgr)

    face_records = extract_features(face_points, processed_rgb, 1)
    #nonface_records = extract_features(nonface_points, processed_rgb, 0)

    data.extend(face_records)
    #data.extend(nonface_records)

# ===== 寫入 training_raw.txt =====
with open(output_file, "a") as f:
    for label, Y, Cb, Cr in data:
        f.write(f"{label} {Y:.6f} {Cb:.6f} {Cr:.6f}\n")

print("Raw training data appended to", output_file)