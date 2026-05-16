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

            score = (
                e1["value"] + e2["value"] + m["value"]
                - 0.5 * abs(angle_ab_cd - 90)
                - 0.5 * angle_cd_e1
                - 5.0 * abs(ratio - 0.7)
                - 0.2 * dist_to_axis
            )

            valid_triplets.append({
                "left_eye": e1,
                "right_eye": e2,
                "mouth": m,
                "angle_ab_cd": angle_ab_cd,
                "angle_cd_e1": angle_cd_e1,
                "ratio": ratio,
                "dist_to_axis": dist_to_axis,
                "score": score
            })

    valid_triplets = sorted(
        valid_triplets,
        key=lambda x: x["score"],
        reverse=True
    )

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
threshold_e, threshold_m = 1.5, 1e10
m, n, _ = img.shape
img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
skin = predict(image_path, "checkpoints/unet_skin_best.pth")
ellipse, all_points = ellipse_matching(skin)
Eyemap = eyemap(img_rgb)
Mouthmap = mouthmap(img_rgb,all_points)
flat = Mouthmap.flatten()
top_idx = np.argpartition(flat, -10)[-10:]   # 比 sort 快
ys, xs = np.unravel_index(top_idx, Mouthmap.shape)

# -------- 畫圖 --------
plt.figure(figsize=(6, 6))
im = plt.imshow(Mouthmap, cmap='jet')
plt.axis('off')

# 標記點（紅色）
plt.scatter(xs, ys, c='red', s=40, marker='o', edgecolors='black')

plt.colorbar(im, fraction=0.046, pad=0.04)
plt.title("Top-10 Mouthmap Responses", y=-0.15)

plt.savefig("mouthmap_result.png", dpi=200, bbox_inches='tight')
#eye_candidates = []
#mouth_candidates = []
print(len(ellipse))
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
    valid = test_triplets_with_geometry(eye_candidates,mouth_candidates,eigvecs, mean, a, b)

    if len(valid) > 0:
        
        best = valid[0]

        img_result = draw_result(img_rgb, best, e)
        output = cv2.cvtColor(img_result, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, output)