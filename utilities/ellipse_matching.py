import cv2
import numpy as np
import matplotlib.pyplot as plt

def draw_ellipse(img, mean, a, b, eigvecs, color=(0,255,0), thickness=2):

    t = np.linspace(0, 2*np.pi, 200)

    ellipse_local = np.column_stack((a*np.cos(t), b*np.sin(t)))

    ellipse_global = ellipse_local @ eigvecs.T
    ellipse_global[:,0] += mean[0]
    ellipse_global[:,1] += mean[1]

    pts = np.round(ellipse_global).astype(np.int32)
    pts = pts.reshape((-1,1,2))

    cv2.polylines(img, [pts], True, color, thickness)

def ellipse_matching(img_binary):
    num_labels, labels = cv2.connectedComponents(img_binary)
    print("Objects found:", num_labels - 1) 

    results = []
    all_points = []

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
        aspect_ratio = a/b

        if aspect_ratio < 3 and aspect_ratio > 0.33 and ratio > 0.7:    
            results.append({
                "label_id": label_id,
                "center": mean,
                "a": a,
                "b": b,
                "points":points,
                "eigvecs": eigvecs
            })

            all_points.append(points)
    if len(all_points) > 0:
        all_points = np.vstack(all_points)
    else:
        all_points = np.empty((0, 2), dtype=int)

    return results,all_points

        