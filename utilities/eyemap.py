import cv2
from matplotlib import pyplot as plt
from scipy.signal import convolve2d
import numpy as np

def edge_filter(sigma=1, l=10):
    filter = np.zeros(2*l+1)
    for i in range(10):
        filter[i]=np.exp(sigma*(i-10))
        filter[i+l+1]=np.exp(-1*sigma*(i+1))
    C = np.sum(filter[:l])
    filter = filter/C
    return filter

def eyemap(img_rgb, k=3, l=10):
    img_ycrcb = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YCrCb)
    Y = img_ycrcb[:, :, 0]
    kernel = np.array([
        [0,1,0],
        [1,1,1],
        [0,1,0]
    ], dtype=np.uint8)
    for i in range(k):
        Y = cv2.erode(Y, kernel)
    Y1 = Y/255
    eyemapl = (1-Y1)/(1+l*Y1)

    Cr = img_ycrcb[:, :, 1]+127.5
    Cb = img_ycrcb[:, :, 2]+127.5
    Cr_inv = Cr.max() - Cr
    eyemapc = (Cb**2 + Cr_inv**2 + Cb * Cr_inv) / 3

    f1 = edge_filter(sigma=1)
    f2 = edge_filter(sigma=0.2)
    f3 = edge_filter(sigma=0.05)

    Y_1x = convolve2d(Y,f1.reshape(1,-1),'same')
    Y_2x = convolve2d(Y,f2.reshape(1,-1),'same')
    Y_3x = convolve2d(Y,f3.reshape(1,-1),'same')
    Y_1y = convolve2d(Y,f1.reshape(-1,1),'same')
    Y_2y = convolve2d(Y,f2.reshape(-1,1),'same')
    Y_3y = convolve2d(Y,f3.reshape(-1,1),'same')

    eyemapt = np.maximum.reduce([
        np.abs(Y_1x),
        np.abs(Y_2x),
        np.abs(Y_3x),
        np.abs(Y_1y),
        np.abs(Y_2y),
        np.abs(Y_3y)
    ])

    eyemapl = (eyemapl-eyemapl.mean())/eyemapl.std()
    eyemapc = (eyemapc-eyemapc.mean())/eyemapc.std()
    eyemapt = (eyemapt-eyemapt.mean())/eyemapt.std()

    eyemap_result = 0.45*eyemapl + 0.45*eyemapc + 0.1*eyemapt

    return eyemap_result

if __name__ ==  "__main__":
    image = cv2.imread("TestImagesForPrograms/19.jpg")
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    eye_map = eyemap(img_rgb)

    # -------- 找 top 10 --------
    flat = eye_map.flatten()
    top_idx = np.argpartition(flat, -10)[-10:]   # 比 sort 快
    ys, xs = np.unravel_index(top_idx, eye_map.shape)

    # -------- 畫圖 --------
    plt.figure(figsize=(6, 6))
    im = plt.imshow(eye_map, cmap='jet')
    plt.axis('off')

    # 標記點（紅色）
    plt.scatter(xs, ys, c='red', s=40, marker='o', edgecolors='black')

    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.title("Top-10 Eyemap Responses", y=-0.15)

    plt.savefig("eyemap_result.png", dpi=200, bbox_inches='tight')
    