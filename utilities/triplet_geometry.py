# utilities/triplet_geometry.py

import numpy as np
from itertools import combinations


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

        if A[0] > B[0]:
            A, B = B, A
            e1, e2 = e2, e1

        D = (A + B) / 2
        AB = B - A
        len_AB = np.linalg.norm(AB)

        if len_AB == 0:
            continue

        if not (0.20 * b <= len_AB <= 1.80 * b):
            continue

        center_to_D = D - ellipse_center
        proj_len = np.dot(center_to_D, major_axis)
        proj_point = ellipse_center + proj_len * major_axis
        dist_to_axis = np.linalg.norm(D - proj_point)

        if dist_to_axis > 0.35 * b:
            continue

        for mouth in mouth_candidates:
            C = np.array([mouth["x"], mouth["y"]], dtype=float)

            CD = D - C
            len_CD = np.linalg.norm(CD)

            if len_CD == 0:
                continue

            cos_cd_e1 = np.dot(CD, major_axis) / (
                len_CD * np.linalg.norm(major_axis) + 1e-8
            )
            cos_cd_e1 = np.clip(cos_cd_e1, -1.0, 1.0)
            angle_cd_e1 = np.degrees(np.arccos(abs(cos_cd_e1)))

            if angle_cd_e1 > 35:
                continue

            cos_ab_cd = np.dot(AB, CD) / (len_AB * len_CD + 1e-8)
            cos_ab_cd = np.clip(cos_ab_cd, -1.0, 1.0)
            angle_ab_cd = np.degrees(np.arccos(abs(cos_ab_cd)))

            if abs(angle_ab_cd - 90) > 25:
                continue

            ratio = len_AB / len_CD

            if not (0.3 <= ratio <= 1.5):
                continue

            valid_triplets.append({
                "left_eye": e1,
                "right_eye": e2,
                "mouth": mouth,
                "angle_ab_cd": angle_ab_cd,
                "angle_cd_e1": angle_cd_e1,
                "ratio": ratio,
                "dist_to_axis": dist_to_axis
            })

    return valid_triplets