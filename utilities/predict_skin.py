# utilities/predict_skin.py

import cv2
import torch
import numpy as np
import sys
import os
from models.unet import UNet

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def predict(image_path, weight_path):
    """
    Compute the skin mask based on chrominance components.

    Args:
        image_path: Path to the input image.
        weight_path: Path to the pre-trained model weights.

    Returns:
        mask: Binary image highlighting possible skin regions.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = UNet().to(device)
    model.load_state_dict(torch.load(weight_path, map_location=device, weights_only=True))
    model.eval()

    img = cv2.imread(image_path)
    h, w = img.shape[:2]

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(img_rgb, (256, 256))

    x = resized.astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))
    x = torch.tensor(x).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(x)
        prob = torch.sigmoid(pred)
        mask = (prob > 0.5).float()

    mask = mask.squeeze().cpu().numpy()
    mask = cv2.resize(mask, (w, h))
    mask = (mask * 255).astype(np.uint8)
    
    return mask


