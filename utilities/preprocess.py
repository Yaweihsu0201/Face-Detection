import cv2
import numpy as np

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