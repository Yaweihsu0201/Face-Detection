import pandas as pd
import cv2
import os

csv_path = "triplet_dataset.csv"
df = pd.read_csv(csv_path)

if "label" not in df.columns:
    df["label"] = -1

save_dir = "label_preview"
os.makedirs(save_dir, exist_ok=True)

def find_image_path(image_name):
    image_name = str(image_name)

    possible_paths = [
        image_name,
        os.path.join("TestImagesForPrograms", image_name),
        os.path.join("TestImagesForPrograms", image_name + ".jpg"),
        os.path.join("TestImagesForPrograms", image_name + ".jpeg"),
        os.path.join("dataset", image_name),
        os.path.join("dataset", image_name + ".jpg"),
        os.path.join("dataset", image_name + ".jpeg"),
    ]

    for p in possible_paths:
        if os.path.exists(p):
            return p

    return None

def draw_triplet_on_image(img, row):
    img_draw = img.copy()

    left_eye = (int(row["left_eye_x"]), int(row["left_eye_y"]))
    right_eye = (int(row["right_eye_x"]), int(row["right_eye_y"]))
    mouth = (int(row["mouth_x"]), int(row["mouth_y"]))

    cv2.circle(img_draw, left_eye, 6, (255, 0, 0), -1)
    cv2.circle(img_draw, right_eye, 6, (0, 255, 0), -1)
    cv2.circle(img_draw, mouth, 6, (0, 0, 255), -1)

    cv2.line(img_draw, left_eye, right_eye, (255, 255, 255), 2)

    mid = (
        int((left_eye[0] + right_eye[0]) / 2),
        int((left_eye[1] + right_eye[1]) / 2)
    )
    cv2.line(img_draw, mid, mouth, (255, 255, 255), 2)

    return img_draw

for idx in range(len(df)):
    if df.loc[idx, "label"] in [0, 1]:
        continue

    row = df.loc[idx]

    image_path = find_image_path(row["image"])
    if image_path is None:
        print(f"Image not found: {row['image']}")
        continue

    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not read image: {image_path}")
        continue

    preview = draw_triplet_on_image(img, row)

    preview_path = os.path.join(save_dir, "current_triplet.png")
    cv2.imwrite(preview_path, preview)

    print("\n--------------------------------")
    print(f"Index: {idx + 1}/{len(df)}")
    print(f"Image: {row['image']}")
    print(f"Preview saved to: {preview_path}")
    print("Open this image and label it.")
    print("1 = valid, 0 = invalid, s = skip, q = quit and save")

    ans = input("Your label: ").strip().lower()

    if ans == "1":
        df.loc[idx, "label"] = 1
    elif ans == "0":
        df.loc[idx, "label"] = 0
    elif ans == "s":
        continue
    elif ans == "q":
        break
    else:
        print("Invalid input, skipped.")
        continue

    df.to_csv(csv_path, index=False)
    print("Saved.")

df.to_csv(csv_path, index=False)
print(f"Final saved to {csv_path}")