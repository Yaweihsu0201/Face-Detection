from utilities.skin_filter import skinfilter 
from utilities.ellipse_matching import ellipse_matching
from utilities.ellipse_matching import draw_ellipse
from utilities.eyemap import eyemap
from utilities.mouthmap import mouthmap
from utilities.preprocess import gamma_correction
import cv2 
from matplotlib import pyplot as plt
import numpy as np
from scipy.signal import convolve2d
from itertools import combinations
import sys

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


def create_circular_kernel(h):
    r = int(h / 40)
    y, x = np.ogrid[-r:r+1, -r:r+1]

    mask = x*x + y*y <= r*r

    kernel = np.zeros((2*r+1, 2*r+1), dtype=np.float32)
    kernel[mask] = (40 / h)**2

    return kernel

def create_ellipse_kernel(h, w):
    a = h / 25   # vertical radius
    b = w / 4    # horizontal radius

    hh = int(np.ceil(a))
    ww = int(np.ceil(b))

    y, x = np.ogrid[-hh:hh+1, -ww:ww+1]

    kernel = ((y / a)**2 + (x / b)**2 <= 1).astype(np.float32)

    return kernel

def angle_eye_mouth(A, B, C):
    A = np.array(A, dtype=float)
    B = np.array(B, dtype=float)
    C = np.array(C, dtype=float)

    D = (A + B) / 2

    AB = B - A
    CD = D - C

    norm_AB = np.linalg.norm(AB)
    norm_CD = np.linalg.norm(CD)

    if norm_AB == 0 or norm_CD == 0:
        return None

    cos_theta = np.dot(AB, CD) / (norm_AB * norm_CD)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    theta = np.degrees(np.arccos(cos_theta))
    return theta

def check_cd_parallel_e1(A, B, C, eigvecs, angle_thresh_deg=30):
    A = np.array(A, dtype=float)
    B = np.array(B, dtype=float)
    C = np.array(C, dtype=float)

    D = (A + B) / 2
    CD = D - C
    e1 = eigvecs[:, 0].astype(float)

    norm_CD = np.linalg.norm(CD)
    norm_e1 = np.linalg.norm(e1)

    if norm_CD == 0 or norm_e1 == 0:
        return False, None

    cos_theta = np.dot(CD, e1) / (norm_CD * norm_e1)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    theta_deg = np.degrees(np.arccos(abs(cos_theta)))

    return theta_deg <= angle_thresh_deg, theta_deg

def check_length_ratio(A, B, C):
    A = np.array(A, dtype=float)
    B = np.array(B, dtype=float)
    C = np.array(C, dtype=float)

    D = (A + B) / 2

    AB = B - A
    CD = D - C

    len_AB = np.linalg.norm(AB)
    len_CD = np.linalg.norm(CD)

    if len_CD == 0:
        return False, None

    ratio = len_AB / len_CD
    ok = 0.4 <= ratio <= 2.5

    return ok, ratio

def test_triplets_with_geometry(eye_candidates, mouth_candidates, eigvecs, center):
    valid_triplets = []
    ellipse_center = np.array(center, dtype=float)
    major_axis = eigvecs[:, 0].astype(float)
    major_axis = major_axis / np.linalg.norm(major_axis)
    for e1, e2 in combinations(eye_candidates, 2):
        A = np.array([e1["x"], e1["y"]], dtype=float)
        B = np.array([e2["x"], e2["y"]], dtype=float)

        if A[0] > B[0]:
            A, B = B, A
            e1, e2 = e2, e1

        D = (A + B) / 2
        AB = B - A
        len_AB = np.linalg.norm(AB)

        if len_AB == 0:
            continue
        center_to_D = D - ellipse_center

        proj_len = np.dot(center_to_D, major_axis)
        proj_point = ellipse_center + proj_len * major_axis

        dist_to_axis = np.linalg.norm(D - proj_point)

        if dist_to_axis > 0.15 * 50:   # 先用固定值也可以，例如 15
            continue
        for m in mouth_candidates:
            C = np.array([m["x"], m["y"]], dtype=float)
            if A[1] >= C[1] or B[1] >= C[1]:
                continue
            CD = D - C
            len_CD = np.linalg.norm(CD)

            if len_CD == 0:
                continue

            cos_ab_cd = np.dot(AB, CD) / (len_AB * len_CD)
            cos_ab_cd = np.clip(cos_ab_cd, -1.0, 1.0)
            angle_ab_cd = np.degrees(np.arccos(cos_ab_cd))

            if abs(angle_ab_cd - 90) > 20:
                continue

            major_axis = eigvecs[:, 0].astype(float)
            cos_cd_e1 = np.dot(CD, major_axis) / (
                len_CD * np.linalg.norm(major_axis)
            )
            cos_cd_e1 = np.clip(cos_cd_e1, -1.0, 1.0)
            angle_cd_e1 = np.degrees(np.arccos(abs(cos_cd_e1)))

            if angle_cd_e1 > 30:
                continue

            ratio = len_AB / len_CD
            if not (0.4 <= ratio <= 1.0):
                continue

            valid_triplets.append({
                "left_eye": e1,
                "right_eye": e2,
                "mouth": m,
                "angle_ab_cd": angle_ab_cd,
                "angle_cd_e1": angle_cd_e1,
                "ratio": ratio
            })

    return valid_triplets

def draw_result(img_rgb, result, ellipse_info):
    img_draw = img_rgb.copy()

    left_eye = result["left_eye"]
    right_eye = result["right_eye"]
    mouth = result["mouth"]

    center = ellipse_info["center"]
    a = ellipse_info["a"]
    b = ellipse_info["b"]
    eigvecs = ellipse_info["eigvecs"]

    cv2.circle(img_draw, (int(left_eye["x"]), int(left_eye["y"])), 5, (255, 0, 0), -1)
    cv2.circle(img_draw, (int(right_eye["x"]), int(right_eye["y"])), 5, (0, 255, 0), -1)
    cv2.circle(img_draw, (int(mouth["x"]), int(mouth["y"])), 5, (255, 255, 0), -1)

    cv2.line(
        img_draw,
        (int(left_eye["x"]), int(left_eye["y"])),
        (int(right_eye["x"]), int(right_eye["y"])),
        (255, 255, 255),
        2
    )

    dx = (left_eye["x"] + right_eye["x"]) / 2
    dy = (left_eye["y"] + right_eye["y"]) / 2
    cv2.line(
        img_draw,
        (int(dx), int(dy)),
        (int(mouth["x"]), int(mouth["y"])),
        (255, 255, 255),
        2
    )

    major_axis = eigvecs[:, 0]
    angle = np.degrees(np.arctan2(major_axis[1], major_axis[0]))

    cv2.ellipse(
        img_draw,
        center=(int(center[0]), int(center[1])),
        axes=(int(a), int(b)),
        angle=float(angle),
        startAngle=0,
        endAngle=360,
        color=(255, 0, 255),
        thickness=2
    )

    return img_draw

image_path = sys.argv[1]
output_path = sys.argv[2]

img = cv2.imread(image_path)
img_E = preprocess_image(img)
threshold_e, threshold_m = 2, 1e10
m, n, _ = img.shape
img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
img_rgb_E = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
skin = skinfilter(img_rgb_E)
ellipse, all_points = ellipse_matching(skin)
Eyemap = eyemap(img_rgb)
Mouthmap = mouthmap(img_rgb,all_points)
#eye_candidates = []
#mouth_candidates = []
for e in ellipse:
    eye_candidates = []
    mouth_candidates = []
    a,b = e["a"],e["b"]
    h = max(a,b)
    w = min(a,b)
    mean = e["center"]
    eigvecs = e["eigvecs"]
    kernel_c = create_circular_kernel(h)
    kernel_e = create_ellipse_kernel(h,w)
    eyemap_conv = convolve2d(Eyemap, kernel_c, mode='same')
    mouthmap_conv = convolve2d(Mouthmap, kernel_e, mode='same')
    for i in range(1,m-1):
        for j in range(1,n-1):
            val = eyemap_conv[i][j]

            if val <= threshold_e:
                continue

            patch = eyemap_conv[i-1:i+2, j-1:j+2]
            if val < np.max(patch):
                continue

            p = np.array([j, i], dtype=np.float32)
            z = p - mean
            xe = z @ eigvecs   # [x1, x2]

            x1, x2 = xe[0], xe[1]

            if (x1**2) / (a**2) + (x2**2) / (b**2) > 0.8:
                continue

            eye_candidates.append({
                "m": i,
                "n": j,
                "x": j,
                "y": i,
                "value": val,
                "x1": x1,
                "x2": x2
            })
            print(val)
    for i in range(1,m-1):
        for j in range(1,n-1):
            val = mouthmap_conv[i][j]

            if val <= threshold_m:
                continue

            patch = mouthmap_conv[i-1:i+2, j-1:j+2]
            if val < np.max(patch):
                continue

            p = np.array([j, i], dtype=np.float32)
            z = p - mean
            xe = z @ eigvecs   # [x1, x2]

            x1, x2 = xe[0], xe[1]

            if (x1**2) / (a**2) + (x2**2) / (b**2) > 0.8:
                continue

            mouth_candidates.append({
                "m": i,
                "n": j,
                "x": j,
                "y": i,
                "value": val,
                "x1": x1,
                "x2": x2
            })
    if len(eye_candidates)<2 or len(mouth_candidates)<1:
        continue
    valid = test_triplets_with_geometry(eye_candidates,mouth_candidates,eigvecs, mean)
    if len(valid) > 0:
        best = valid[0]

        img_result = draw_result(img_rgb, best, e)
        output = cv2.cvtColor(img_result, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, output)