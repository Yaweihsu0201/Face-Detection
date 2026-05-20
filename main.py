import cv2 
import numpy as np
from scipy.signal import convolve2d
from itertools import combinations
import sys
import joblib
import os
import pandas as pd
import argparse

from utilities.predict_skin import predict
from utilities.ellipse_matching import ellipse_matching
from utilities.eyemap import eyemap
from utilities.mouthmap import mouthmap
from utilities.triplet_finder import find_best_triplet_over_ellipses

model = joblib.load("checkpoints/triplet_scorer_rf.pkl")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to input image")
    parser.add_argument("--output", required=True, help="Path to save output image")
    return parser.parse_args()


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

def main():
    args = parse_args()

    image_path = args.input
    output_path = args.output

    img = cv2.imread(image_path)
    threshold_e, threshold_m = 1.5, 1e10
    m, n, _ = img.shape
    img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    skin = predict(image_path, "checkpoints/unet_skin_best.pth")
    ellipse, all_points = ellipse_matching(skin)
    Eyemap = eyemap(img_rgb)
    Mouthmap = mouthmap(img_rgb,all_points)

    best_result = find_best_triplet_over_ellipses(
        ellipses=ellipse,
        eye_map=Eyemap,
        mouth_map=Mouthmap,
        image_shape=img_rgb.shape,
        model=model,
        geometry_func=test_triplets_with_geometry,
        threshold_e=threshold_e,
        threshold_m=threshold_m
    )

    best_triplet = best_result["best_triplet"]
    best_score = best_result["best_score"]
    best_ellipse_info = best_result["best_ellipse_info"]
    print("modified!")
    # Save the highest scoring triplet image
    if best_triplet is not None:
        img_best = draw_result(img_rgb, best_triplet, best_ellipse_info)  
        # Convert to BGR for cv2.imwrite
        img_best_bgr = cv2.cvtColor(img_best, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, img_best_bgr)
    else:
        print("No valid triplets found in the image.")

if __name__ == "__main__":
    main()