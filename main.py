import cv2 
import numpy as np
import joblib
import argparse

from utilities.predict_skin import predict
from utilities.ellipse_matching import ellipse_matching
from utilities.eyemap import eyemap
from utilities.mouthmap import mouthmap
from utilities.triplet_finder import find_best_triplet_over_ellipses
from utilities.visualization import draw_result

model = joblib.load("checkpoints/triplet_scorer_rf.pkl")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to input image")
    parser.add_argument("--output", required=True, help="Path to save output image")
    return parser.parse_args()

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
        threshold_e=threshold_e,
        threshold_m=threshold_m
    )

    best_triplet = best_result["best_triplet"]
    best_ellipse_info = best_result["best_ellipse_info"]

    if best_triplet is not None:
        img_best = draw_result(img_rgb, best_triplet, best_ellipse_info)  
        img_best_bgr = cv2.cvtColor(img_best, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, img_best_bgr)
    else:
        print("No valid triplets found in the image.")

if __name__ == "__main__":
    main()