import cv2
import matplotlib.pyplot as plt
import numpy as np

sample_name_1 = "TestImagesForPrograms/1.jpg"
sample_name_2 = "TestImagesForPrograms/2.jpg"
sample_name_3 = "TestImagesForPrograms/3.jpg"
sample_name_4 = "TestImagesForPrograms/4.jpg"
sample_name_5 = "TestImagesForPrograms/5.jpg"
sample_name_6 = "TestImagesForPrograms/6.jpg"

sample_list = [sample_name_1,sample_name_2,sample_name_3,sample_name_4,sample_name_5,sample_name_6]
adding_list = ["TestImagesForPrograms/12.jpg", "TestImagesForPrograms/15.jpg", "TestImagesForPrograms/16.jpg", "TestImagesForPrograms/18.jpg"]

def collect_points(n, title, img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.title(title)
    plt.axis("on")

    pts = plt.ginput(n, timeout=0)
    plt.show()

    return pts


def extract_features(points, width, height, label):
    records = []

    for x, y in points:

        px = int(round(x))
        py = int(round(y))

        px = max(0, min(px, width - 1))
        py = max(0, min(py, height - 1))

        r, g, b = img_rgb[py, px]

        Y  =  0.299*r + 0.587*g + 0.114*b
        Cb = -0.168736*r - 0.331264*g + 0.5*b
        Cr =  0.5*r - 0.418688*g - 0.081312*b

        records.append([label, Y, Cb, Cr])

    return records

data = []

# --- collect data ---
for element in sample_list:
    sample = cv2.imread(element)
    img_rgb = cv2.cvtColor(sample, cv2.COLOR_BGR2RGB)
    height, width, _ = img_rgb.shape

    face_points = collect_points(10, "Click 10 FACE points", sample)
    nonface_points = collect_points(10, "Click 10 NON-FACE points", sample)

    face_records = extract_features(face_points, width, height, 1)
    nonface_records = extract_features(nonface_points, width, height, 0)

    data = data + face_records + nonface_records

"""
for element in adding_list:
    sample = cv2.imread(element)
    img_rgb = cv2.cvtColor(sample, cv2.COLOR_BGR2RGB)
    height, width, _ = img_rgb.shape
    nonface_points = collect_points(10, "Click 10 NON-FACE points", sample)
    nonface_records = extract_features(nonface_points, width, height, 0)

    data = data + nonface_records
"""

"""
# --- normalization ---

data = np.array(data)
features = data[:, 1:]

mean = features.mean(axis=0)
std = features.std(axis=0)

features_norm = (features - mean) / std
data[:, 1:] = features_norm

np.save("model_3/mean.npy", mean)
np.save("model_3/std.npy", std)

#np.random.shuffle(data)
"""

# --- write training file ---
output_file = "model/training_raw.txt"

with open(output_file, "w") as f:
    for label, Y, Cb, Cr in data:
        f.write(f"{label} {Y:.6f} {Cb:.6f} {Cr:.6f}\n")

print("Raw training data appended")
