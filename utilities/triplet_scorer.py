# utilities/triplet_scorer.py

import numpy as np
import pandas as pd


FEATURE_NAMES = [
    "e1_e2_dist_norm",
    "e1_m_dist_norm",
    "e2_m_dist_norm",
    "midpoint_m_dist_norm",

    "left_eye_x1_norm",
    "left_eye_x2_norm",
    "right_eye_x1_norm",
    "right_eye_x2_norm",
    "mouth_x1_norm",
    "mouth_x2_norm",

    "left_eye_center_dist_norm",
    "right_eye_center_dist_norm",
    "mouth_center_dist_norm",

    "angle_eyes_mouth",
    "angle_to_major_axis",

    "ellipse_ab_ratio",

    "major_axis_x",
    "major_axis_y",
    "minor_axis_x",
    "minor_axis_y",

    "left_eye_value",
    "right_eye_value",
    "mouth_value",
]


def extract_triplet_features(e1, e2, m, ellipse_info):
    center = ellipse_info["center"]
    a = ellipse_info["a"]
    b = ellipse_info["b"]
    eigvecs = ellipse_info["eigvecs"]

    major_axis = eigvecs[:, 0]
    minor_axis = eigvecs[:, 1]

    ellipse_size = np.sqrt(a**2 + b**2)

    e1_pos = np.array([e1["x"] - center[0], e1["y"] - center[1]], dtype=float)
    e2_pos = np.array([e2["x"] - center[0], e2["y"] - center[1]], dtype=float)
    m_pos = np.array([m["x"] - center[0], m["y"] - center[1]], dtype=float)

    e1_e2_dist = np.linalg.norm(e2_pos - e1_pos) / (ellipse_size + 1e-8)
    e1_m_dist = np.linalg.norm(m_pos - e1_pos) / (ellipse_size + 1e-8)
    e2_m_dist = np.linalg.norm(m_pos - e2_pos) / (ellipse_size + 1e-8)

    eyes_midpoint = (e1_pos + e2_pos) / 2
    midpoint_m_dist = np.linalg.norm(m_pos - eyes_midpoint) / (ellipse_size + 1e-8)

    eyes_mouth_vec = m_pos - eyes_midpoint
    eyes_eyes_vec = e2_pos - e1_pos

    if np.linalg.norm(eyes_eyes_vec) > 1e-8 and np.linalg.norm(eyes_mouth_vec) > 1e-8:
        cos_angle = np.dot(eyes_eyes_vec, eyes_mouth_vec) / (
            np.linalg.norm(eyes_eyes_vec) * np.linalg.norm(eyes_mouth_vec) + 1e-8
        )
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle_eyes_mouth = np.degrees(np.arccos(abs(cos_angle)))
    else:
        angle_eyes_mouth = 0.0

    if np.linalg.norm(eyes_mouth_vec) > 1e-8:
        cos_angle_major = np.dot(eyes_mouth_vec, major_axis) / (
            np.linalg.norm(eyes_mouth_vec) + 1e-8
        )
        cos_angle_major = np.clip(cos_angle_major, -1.0, 1.0)
        angle_to_major_axis = np.degrees(np.arccos(abs(cos_angle_major)))
    else:
        angle_to_major_axis = 0.0

    e1_x1, e1_x2 = e1["x1"], e1["x2"]
    e2_x1, e2_x2 = e2["x1"], e2["x2"]
    m_x1, m_x2 = m["x1"], m["x2"]

    e1_x1_norm = e1_x1 / (a + 1e-8)
    e1_x2_norm = e1_x2 / (b + 1e-8)
    e2_x1_norm = e2_x1 / (a + 1e-8)
    e2_x2_norm = e2_x2 / (b + 1e-8)
    m_x1_norm = m_x1 / (a + 1e-8)
    m_x2_norm = m_x2 / (b + 1e-8)

    e1_center_dist = np.sqrt(e1_x1**2 + e1_x2**2) / (ellipse_size + 1e-8)
    e2_center_dist = np.sqrt(e2_x1**2 + e2_x2**2) / (ellipse_size + 1e-8)
    m_center_dist = np.sqrt(m_x1**2 + m_x2**2) / (ellipse_size + 1e-8)

    return [
        e1_e2_dist,
        e1_m_dist,
        e2_m_dist,
        midpoint_m_dist,

        e1_x1_norm,
        e1_x2_norm,
        e2_x1_norm,
        e2_x2_norm,
        m_x1_norm,
        m_x2_norm,

        e1_center_dist,
        e2_center_dist,
        m_center_dist,

        angle_eyes_mouth,
        angle_to_major_axis,

        a / (b + 1e-8),

        major_axis[0],
        major_axis[1],
        minor_axis[0],
        minor_axis[1],

        e1["value"],
        e2["value"],
        m["value"],
    ]


def score_triplets(valid_triplets, ellipse_info, model):
    if len(valid_triplets) == 0:
        return []

    features_list = []

    for triplet in valid_triplets:
        features = extract_triplet_features(
            triplet["left_eye"],
            triplet["right_eye"],
            triplet["mouth"],
            ellipse_info
        )
        features_list.append(features)

    X = pd.DataFrame(features_list, columns=FEATURE_NAMES)
    scores = model.predict_proba(X)[:, 1]

    for triplet, score in zip(valid_triplets, scores):
        triplet["score"] = float(score)

    valid_triplets = sorted(
        valid_triplets,
        key=lambda x: x["score"],
        reverse=True
    )

    return valid_triplets