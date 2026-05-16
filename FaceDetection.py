from utilities.predict_skin import predict
from utilities.ellipse_matching import ellipse_matching
from utilities.ellipse_matching import draw_ellipse
from utilities.eyemap import eyemap
from utilities.mouthmap import mouthmap
import cv2 
from matplotlib import pyplot as plt
import numpy as np
from scipy.signal import convolve2d
from itertools import combinations
import sys
import joblib
import os
import pandas as pd
import math
model = joblib.load("checkpoints/triplet_scorer_rf.pkl")

def extract_triplet_features(e1, e2, m, ellipse_info):
    center = ellipse_info["center"]
    a = ellipse_info["a"]
    b = ellipse_info["b"]
    eigvecs = ellipse_info["eigvecs"]

    major_axis = eigvecs[:, 0]
    minor_axis = eigvecs[:, 1]

    # Ellipse size (for normalization)
    ellipse_size = np.sqrt(a**2 + b**2)

    # Convert absolute coordinates to relative (centered at ellipse center)
    e1_pos = np.array([e1["x"] - center[0], e1["y"] - center[1]], dtype=float)
    e2_pos = np.array([e2["x"] - center[0], e2["y"] - center[1]], dtype=float)
    m_pos = np.array([m["x"] - center[0], m["y"] - center[1]], dtype=float)

    # Relative distances (normalized by ellipse size)
    e1_e2_dist = np.linalg.norm(e2_pos - e1_pos) / (ellipse_size + 1e-8)
    e1_m_dist = np.linalg.norm(m_pos - e1_pos) / (ellipse_size + 1e-8)
    e2_m_dist = np.linalg.norm(m_pos - e2_pos) / (ellipse_size + 1e-8)

    # Eyes midpoint
    eyes_midpoint = (e1_pos + e2_pos) / 2
    midpoint_m_dist = np.linalg.norm(m_pos - eyes_midpoint) / (ellipse_size + 1e-8)

    # Eyes to mouth vector
    eyes_mouth_vec = m_pos - eyes_midpoint
    eyes_eyes_vec = e2_pos - e1_pos

    # Angles between vectors (relative geometry)
    if np.linalg.norm(eyes_eyes_vec) > 1e-8 and np.linalg.norm(eyes_mouth_vec) > 1e-8:
        cos_angle = np.dot(eyes_eyes_vec, eyes_mouth_vec) / (
            np.linalg.norm(eyes_eyes_vec) * np.linalg.norm(eyes_mouth_vec) + 1e-8
        )
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle_eyes_mouth = np.degrees(np.arccos(abs(cos_angle)))
    else:
        angle_eyes_mouth = 0.0

    # Angle between eyes-mouth vector and major axis
    if np.linalg.norm(eyes_mouth_vec) > 1e-8:
        cos_angle_major = np.dot(eyes_mouth_vec, major_axis) / (
            np.linalg.norm(eyes_mouth_vec) + 1e-8
        )
        cos_angle_major = np.clip(cos_angle_major, -1.0, 1.0)
        angle_to_major_axis = np.degrees(np.arccos(abs(cos_angle_major)))
    else:
        angle_to_major_axis = 0.0

    # Relative positions in ellipse coordinates
    e1_x1, e1_x2 = e1["x1"], e1["x2"]
    e2_x1, e2_x2 = e2["x1"], e2["x2"]
    m_x1, m_x2 = m["x1"], m["x2"]

    # Normalize by ellipse axes
    e1_x1_norm = e1_x1 / (a + 1e-8)
    e1_x2_norm = e1_x2 / (b + 1e-8)
    e2_x1_norm = e2_x1 / (a + 1e-8)
    e2_x2_norm = e2_x2 / (b + 1e-8)
    m_x1_norm = m_x1 / (a + 1e-8)
    m_x2_norm = m_x2 / (b + 1e-8)

    # Distance from ellipse center (normalized)
    e1_center_dist = np.sqrt(e1_x1**2 + e1_x2**2) / (ellipse_size + 1e-8)
    e2_center_dist = np.sqrt(e2_x1**2 + e2_x2**2) / (ellipse_size + 1e-8)
    m_center_dist = np.sqrt(m_x1**2 + m_x2**2) / (ellipse_size + 1e-8)

    features = [
        # Relative distances (normalized by ellipse size)
        e1_e2_dist,
        e1_m_dist,
        e2_m_dist,
        midpoint_m_dist,

        # Ellipse-normalized coordinates (relative to ellipse)
        e1_x1_norm,
        e1_x2_norm,
        e2_x1_norm,
        e2_x2_norm,
        m_x1_norm,
        m_x2_norm,

        # Distance from center (normalized)
        e1_center_dist,
        e2_center_dist,
        m_center_dist,

        # Angles (relative geometry)
        angle_eyes_mouth,
        angle_to_major_axis,

        # Ellipse shape (relative)
        a / (b + 1e-8),

        # Ellipse orientation
        major_axis[0],
        major_axis[1],
        minor_axis[0],
        minor_axis[1],

        # Map response values
        e1["value"],
        e2["value"],
        m["value"],
    ]

    return features

FEATURE_NAMES = [
    # Relative distances (normalized by ellipse size)
    "e1_e2_dist_norm",
    "e1_m_dist_norm",
    "e2_m_dist_norm",
    "midpoint_m_dist_norm",

    # Ellipse-normalized coordinates (relative to ellipse)
    "left_eye_x1_norm",
    "left_eye_x2_norm",
    "right_eye_x1_norm",
    "right_eye_x2_norm",
    "mouth_x1_norm",
    "mouth_x2_norm",

    # Distance from ellipse center (normalized)
    "left_eye_center_dist_norm",
    "right_eye_center_dist_norm",
    "mouth_center_dist_norm",

    # Angles (relative geometry)
    "angle_eyes_mouth",
    "angle_to_major_axis",

    # Ellipse shape (relative)
    "ellipse_ab_ratio",

    # Ellipse orientation
    "major_axis_x",
    "major_axis_y",
    "minor_axis_x",
    "minor_axis_y",

    # Map response values
    "left_eye_value",
    "right_eye_value",
    "mouth_value",
]

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

def test_triplets_with_geometry(
    eye_candidates,
    mouth_candidates,
    eigvecs,
    center,
    a,
    b
):
    valid_triplets = []

    ellipse_center = np.array(center, dtype=float)

    major_axis = eigvecs[:, 0].astype(float)
    major_axis = major_axis / (np.linalg.norm(major_axis) + 1e-8)

    for e1, e2 in combinations(eye_candidates, 2):
        A = np.array([e1["x"], e1["y"]], dtype=float)
        B = np.array([e2["x"], e2["y"]], dtype=float)

        # 讓 A 永遠是左眼，B 永遠是右眼
        if A[0] > B[0]:
            A, B = B, A
            e1, e2 = e2, e1

        D = (A + B) / 2
        AB = B - A
        len_AB = np.linalg.norm(AB)

        if len_AB == 0:
            continue

        # 雙眼距離限制：避免兩點太近或太遠
        if not (0.20 * b <= len_AB <= 1.80 * b):
            continue

        # 雙眼 midpoint 要接近臉的長軸
        center_to_D = D - ellipse_center
        proj_len = np.dot(center_to_D, major_axis)
        proj_point = ellipse_center + proj_len * major_axis
        dist_to_axis = np.linalg.norm(D - proj_point)

        if dist_to_axis > 0.35 * b:
            continue

        for m in mouth_candidates:
            C = np.array([m["x"], m["y"]], dtype=float)

            CD = D - C
            len_CD = np.linalg.norm(CD)

            if len_CD == 0:
                continue

            # 嘴巴到眼睛中點的方向應該接近臉長軸
            cos_cd_e1 = np.dot(CD, major_axis) / (
                len_CD * np.linalg.norm(major_axis) + 1e-8
            )
            cos_cd_e1 = np.clip(cos_cd_e1, -1.0, 1.0)
            angle_cd_e1 = np.degrees(np.arccos(abs(cos_cd_e1)))

            if angle_cd_e1 > 35:
                continue

            # 雙眼連線 AB 應該接近垂直於 CD
            cos_ab_cd = np.dot(AB, CD) / (len_AB * len_CD + 1e-8)
            cos_ab_cd = np.clip(cos_ab_cd, -1.0, 1.0)
            angle_ab_cd = np.degrees(np.arccos(abs(cos_ab_cd)))

            # 因為用了 abs(cos)，垂直時 angle 接近 90
            if abs(angle_ab_cd - 90) > 25:
                continue

            ratio = len_AB / len_CD

            if not (0.3 <= ratio <= 1.5):
                continue

            valid_triplets.append({
                "left_eye": e1,
                "right_eye": e2,
                "mouth": m,
                "angle_ab_cd": angle_ab_cd,
                "angle_cd_e1": angle_cd_e1,
                "ratio": ratio,
                "dist_to_axis": dist_to_axis
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
output_path = sys.argv[2]
image_path = sys.argv[1]
image_name = os.path.splitext(os.path.basename(image_path))[0]
img = cv2.imread(image_path)
threshold_e, threshold_m = 1.5, 1e10
m, n, _ = img.shape
img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
skin = predict(image_path, "checkpoints/unet_skin_best.pth")
ellipse, all_points = ellipse_matching(skin)
Eyemap = eyemap(img_rgb)
Mouthmap = mouthmap(img_rgb,all_points)

# Track the highest scoring triplet overall
best_triplet = None
best_score = -1.0
best_ellipse_info = None

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

    # First use test_triplets_with_geometry to extract valid triplets
    valid = test_triplets_with_geometry(
        eye_candidates,
        mouth_candidates,
        e["eigvecs"],
        e["center"],
        e["a"],
        e["b"]
    )
    
    # Then use random forest model to predict the score of each triplet
    if len(valid) > 0:
        features_list = []
        for triplet in valid:
            features = extract_triplet_features(
                triplet["left_eye"],
                triplet["right_eye"],
                triplet["mouth"],
                e
            )
            features_list.append(features)
        
        X = pd.DataFrame(features_list, columns=FEATURE_NAMES)
        scores = model.predict_proba(X)[:, 1]
        
        # Update scores with model predictions
        for triplet, score in zip(valid, scores):
            triplet["score"] = float(score)
        
        # Sort by model score
        valid = sorted(
            valid,
            key=lambda x: x["score"],
            reverse=True
        )
        
        # Track the best triplet overall
        if valid[0]["score"] > best_score:
            best_score = valid[0]["score"]
            best_triplet = valid[0]
            best_ellipse_info = e

# Save the highest scoring triplet image
if best_triplet is not None:
    img_best = draw_result(img_rgb, best_triplet, best_ellipse_info)  
    # Convert to BGR for cv2.imwrite
    img_best_bgr = cv2.cvtColor(img_best, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, img_best_bgr)
else:
    print("No valid triplets found in the image.")
