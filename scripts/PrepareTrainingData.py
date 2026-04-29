import numpy as np
import random
import cv2

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
dir = "model5/"

raw_file = dir+"training_raw.txt"
output_file = dir+"training.txt"
mean_file = dir+"mean.npy"
std_file = dir+"std.npy"

#---Read raw data---
data = np.loadtxt(raw_file)

if data.ndim == 1:
    data = data.reshape(1, -1)

#---Hashing---
data = data.tolist()
random.shuffle(data)
data = np.array(data)

#---Seperate features and labels
labels = data[:, 0].astype(int)
features = data[:, 1:]

#---Compute mean and std
mean = features.mean(axis=0)
std = features.std(axis=0)

std[std == 0] = 1.0

#---Normalization---
features_norm = (features - mean) / std

#---Output---
with open(output_file, "w") as f:
    for label, row in zip(labels, features_norm):
        feature_str = " ".join(f"{i+1}:{v:.6f}" for i, v in enumerate(row))
        f.write(f"{label} {feature_str}\n")

np.save(mean_file, mean)
np.save(std_file, std)

print("Prepared training data saved to", output_file)
print("Mean saved to", mean_file)
print("Std saved to", std_file)
print("mean =", mean)
print("std =", std)