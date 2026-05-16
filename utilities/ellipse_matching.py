import cv2
import numpy as np

def draw_ellipse(img, mean, a, b, eigvecs, color=(0,255,0), thickness=2):
    """
    Draw an ellipse on the image based on the given parameters.

    Args:
        img: Input image on which to draw the ellipse.
        mean: Center of the ellipse (x_mean, y_mean).
        a: Semi-major axis length.
        b: Semi-minor axis length.
        eigvecs: Eigenvectors representing the orientation of the ellipse.
        color: Color of the ellipse (default is green).
        thickness: Thickness of the ellipse outline (default is 2).

    Returns:
        Image with the drawn ellipse.
    """
    t = np.linspace(0, 2*np.pi, 200)

    ellipse_local = np.column_stack((a*np.cos(t), b*np.sin(t)))

    ellipse_global = ellipse_local @ eigvecs.T
    ellipse_global[:,0] += mean[0]
    ellipse_global[:,1] += mean[1]

    pts = np.round(ellipse_global).astype(np.int32)
    pts = pts.reshape((-1,1,2))

    cv2.polylines(img, [pts], True, color, thickness)

def ellipse_matching(img_binary):
    """
    detect connected components in the binary image, fit an ellipse to each component, and filter based on shape criteria.

    Args:
        img_binary: Input binary image.

    Returns:
        List of detected ellipses and all points.
    """
    num_labels, labels = cv2.connectedComponents(img_binary)

    results = []
    all_points = []

    for label_id in range(1, num_labels):
        ys, xs = np.where(labels == label_id)
        if len(xs) == 0:
                continue
        
        points = np.column_stack((xs, ys))
        mean = points.mean(axis=0) 
        Z = points - mean

        ZT_Z = Z.T @ Z
        eigvals, eigvecs = np.linalg.eig(ZT_Z)

        idx = np.argsort(eigvals)[::-1]
        eigvals, eigvecs = eigvals[idx], eigvecs[:, idx]

        ZE = Z @ eigvecs

        x1,x2 = ZE[:,0], ZE[:,1]

        m11,m12 = np.mean(np.abs(x1)), np.mean(np.abs(x2))
        m21,m22 = np.mean(x1**2), np.mean(x2**2)

        a1,a2 = 3*np.pi*m11/4, np.sqrt(4*m21)
        b1,b2 = 3*np.pi*m12/4, np.sqrt(4*m22)

        a,b = (a1+a2)/2, (b1+b2)/2

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

        