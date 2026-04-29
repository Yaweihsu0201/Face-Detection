import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys

def draw_ellipse(img, mean, a, b, eigvecs, color=(0,255,0), thickness=2):

    t = np.linspace(0, 2*np.pi, 200)

    ellipse_local = np.column_stack((a*np.cos(t), b*np.sin(t)))

    ellipse_global = ellipse_local @ eigvecs.T
    ellipse_global[:,0] += mean[0]
    ellipse_global[:,1] += mean[1]

    pts = np.round(ellipse_global).astype(np.int32)
    pts = pts.reshape((-1,1,2))

    cv2.polylines(img, [pts], True, color, thickness)

#if __name__ == "main":
image_path = sys.argv[1]
color_image_path = sys.argv[2]
output_path = sys.argv[3]

img_binary = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
img_color = cv2.imread(color_image_path, cv2.IMREAD_COLOR)

if img_binary is None:
    raise ValueError("Image not found")

_, img_binary = cv2.threshold(img_binary, 127, 255, cv2.THRESH_BINARY)
                              
num_labels, labels = cv2.connectedComponents(img_binary)
print("Objects found:", num_labels - 1) 

for label_id in range(1, num_labels):
    ys, xs = np.where(labels == label_id)
#   mean_x = xs.mean()
#   mean_y = ys.mean()
    if len(xs) == 0:
            continue
    
    points = np.column_stack((xs, ys))
    mean = points.mean(axis=0)   # [x_mean, y_mean]
    Z = points - mean

    ZT_Z = Z.T @ Z
    eigvals, eigvecs = np.linalg.eig(ZT_Z)

    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    ZE = Z @ eigvecs

    x1 = ZE[:,0]
    x2 = ZE[:,1]

    m11 = np.mean(np.abs(x1))
    m12 = np.mean(np.abs(x2))

    m21 = np.mean(x1**2)
    m22 = np.mean(x2**2)

    a1 = 3*np.pi*m11/4
    a2 = np.sqrt(4*m21)
    b1 = 3*np.pi*m12/4
    b2 = np.sqrt(4*m22)

    a = (a1+a2)/2
    b = (b1+b2)/2

    if a <= 0 or b <= 0:
        continue

    val = x1**2 / a**2 + x2**2 / b**2
    inside = val <= 1

    inside_count = np.sum(inside)
    total_count = len(x1)
    ratio = inside_count / total_count
    text = f"ratio={ratio:.3f}"
    draw_ellipse(img_color, mean, a, b, eigvecs)
    cv2.putText(
        img_color,
        text,
        (int(round(mean[0])),int(round(mean[1]))),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 255),
        1,
        cv2.LINE_AA
    )
#label_hue = np.uint8(179 * labels / np.max(labels))
#blank_ch = 255 * np.ones_like(label_hue)
#colored = cv2.merge([label_hue, blank_ch, blank_ch])

#colored = cv2.cvtColor(colored, cv2.COLOR_HSV2BGR)
#colored[label_hue == 0] = 0

cv2.imwrite(output_path, img_color)