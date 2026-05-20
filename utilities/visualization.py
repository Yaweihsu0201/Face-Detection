# utilities/visualization.py

import cv2
import numpy as np


def draw_result(img_rgb, result, ellipse_info):
    img_draw = img_rgb.copy()

    left_eye = result["left_eye"]
    right_eye = result["right_eye"]
    mouth = result["mouth"]

    center = ellipse_info["center"]
    a = ellipse_info["a"]
    b = ellipse_info["b"]
    eigvecs = ellipse_info["eigvecs"]

    cv2.circle(
        img_draw,
        (int(left_eye["x"]), int(left_eye["y"])),
        5,
        (255, 0, 0),
        -1
    )

    cv2.circle(
        img_draw,
        (int(right_eye["x"]), int(right_eye["y"])),
        5,
        (0, 255, 0),
        -1
    )

    cv2.circle(
        img_draw,
        (int(mouth["x"]), int(mouth["y"])),
        5,
        (255, 255, 0),
        -1
    )

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