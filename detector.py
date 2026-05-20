# detector.py

import cv2
import joblib

from utilities.predict_skin import predict
from utilities.ellipse_matching import ellipse_matching
from utilities.eyemap import eyemap
from utilities.mouthmap import mouthmap
from utilities.triplet_finder import find_best_triplet_over_ellipses
from utilities.visualization import draw_result


class FaceDetector:
    def __init__(
        self,
        skin_model_path="checkpoints/unet_skin_best.pth",
        triplet_model_path="checkpoints/triplet_scorer_rf.pkl",
        eye_threshold=1.5,
        mouth_threshold=1e10
    ):
        self.skin_model_path = skin_model_path
        self.triplet_model = joblib.load(triplet_model_path)
        self.eye_threshold = eye_threshold
        self.mouth_threshold = mouth_threshold

    def detect(self, image_path):
        img = cv2.imread(image_path)

        if img is None:
            raise ValueError(f"Cannot read image: {image_path}")

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        skin = predict(image_path, self.skin_model_path)
        ellipses, all_points = ellipse_matching(skin)

        eye_map = eyemap(img_rgb)
        mouth_map = mouthmap(img_rgb, all_points)

        best_result = find_best_triplet_over_ellipses(
            ellipses=ellipses,
            eye_map=eye_map,
            mouth_map=mouth_map,
            image_shape=img_rgb.shape,
            model=self.triplet_model,
            threshold_e=self.eye_threshold,
            threshold_m=self.mouth_threshold
        )

        return {
            "image_rgb": img_rgb,
            "best_triplet": best_result["best_triplet"],
            "best_score": best_result["best_score"],
            "best_ellipse_info": best_result["best_ellipse_info"]
        }

    def save_result(self, image_path, output_path):
        result = self.detect(image_path)

        best_triplet = result["best_triplet"]
        best_ellipse_info = result["best_ellipse_info"]

        if best_triplet is None:
            print("No valid triplets found in the image.")
            return False

        img_best = draw_result(
            result["image_rgb"],
            best_triplet,
            best_ellipse_info
        )

        img_best_bgr = cv2.cvtColor(img_best, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, img_best_bgr)

        return True