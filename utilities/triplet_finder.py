# utilities/triplet_finder.py

import numpy as np
from scipy.signal import convolve2d

from utilities.triplet_scorer import score_triplets
from utilities.triplet_geometry import test_triplets_with_geometry


def create_circular_kernel(h):
    r = int(h / 40)

    r = max(r, 1)

    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    mask = x * x + y * y <= r * r

    kernel = np.zeros((2 * r + 1, 2 * r + 1), dtype=np.float32)
    kernel[mask] = (40 / h) ** 2

    return kernel


def create_ellipse_kernel(h, w):
    a = h / 25
    b = w / 4

    hh = int(np.ceil(a))
    ww = int(np.ceil(b))

    hh = max(hh, 1)
    ww = max(ww, 1)

    y, x = np.ogrid[-hh:hh + 1, -ww:ww + 1]

    kernel = ((y / a) ** 2 + (x / b) ** 2 <= 1).astype(np.float32)

    return kernel


def find_eye_candidates(eyemap_conv, ellipse_info, image_shape, threshold_e=1.5):
    m, n = image_shape[:2]
    eye_candidates = []

    a, b = ellipse_info["a"], ellipse_info["b"]
    mean = ellipse_info["center"]
    eigvecs = ellipse_info["eigvecs"]

    for i in range(1, m - 1):
        for j in range(1, n - 1):
            val = eyemap_conv[i][j]

            if val <= threshold_e:
                continue

            patch = eyemap_conv[i - 1:i + 2, j - 1:j + 2]
            if val < np.max(patch):
                continue

            p = np.array([j, i], dtype=np.float32)
            z = p - mean
            xe = z @ eigvecs

            x1, x2 = xe[0], xe[1]

            if (x1 ** 2) / (a ** 2) + (x2 ** 2) / (b ** 2) > 0.8:
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

    return eye_candidates


def find_mouth_candidates(mouthmap_conv, ellipse_info, image_shape, threshold_m=1e10):
    m, n = image_shape[:2]
    mouth_candidates = []

    a, b = ellipse_info["a"], ellipse_info["b"]
    mean = ellipse_info["center"]
    eigvecs = ellipse_info["eigvecs"]

    for i in range(1, m - 1):
        for j in range(1, n - 1):
            val = mouthmap_conv[i][j]

            if val <= threshold_m:
                continue

            patch = mouthmap_conv[i - 1:i + 2, j - 1:j + 2]
            if val < np.max(patch):
                continue

            p = np.array([j, i], dtype=np.float32)
            z = p - mean
            xe = z @ eigvecs

            x1, x2 = xe[0], xe[1]

            if (x1 ** 2) / (a ** 2) + (x2 ** 2) / (b ** 2) > 0.8:
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

    return mouth_candidates


def find_best_triplet_over_ellipses(
    ellipses,
    eye_map,
    mouth_map,
    image_shape,
    model,
    threshold_e=1.5,
    threshold_m=1e10
):
    best_triplet = None
    best_score = -1.0
    best_ellipse_info = None

    for ellipse_info in ellipses:
        a, b = ellipse_info["a"], ellipse_info["b"]
        h = max(a, b)
        w = min(a, b)

        kernel_c = create_circular_kernel(h)
        kernel_e = create_ellipse_kernel(h, w)

        eyemap_conv = convolve2d(eye_map, kernel_c, mode="same")
        mouthmap_conv = convolve2d(mouth_map, kernel_e, mode="same")

        eye_candidates = find_eye_candidates(
            eyemap_conv,
            ellipse_info,
            image_shape,
            threshold_e=threshold_e
        )

        mouth_candidates = find_mouth_candidates(
            mouthmap_conv,
            ellipse_info,
            image_shape,
            threshold_m=threshold_m
        )

        if len(eye_candidates) < 2 or len(mouth_candidates) < 1:
            continue

        valid_triplets = test_triplets_with_geometry(
            eye_candidates,
            mouth_candidates,
            ellipse_info["eigvecs"],
            ellipse_info["center"],
            ellipse_info["a"],
            ellipse_info["b"]
        )

        if len(valid_triplets) == 0:
            continue

        valid_triplets = score_triplets(
            valid_triplets,
            ellipse_info,
            model
        )

        if valid_triplets[0]["score"] > best_score:
            best_score = valid_triplets[0]["score"]
            best_triplet = valid_triplets[0]
            best_ellipse_info = ellipse_info

    return {
        "best_triplet": best_triplet,
        "best_score": best_score,
        "best_ellipse_info": best_ellipse_info
    }