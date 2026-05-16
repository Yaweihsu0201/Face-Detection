import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


FEATURE_NAMES = [
    # Relative distances (normalized by ellipse size)
    "e1_e2_dist_norm",
    "e1_m_dist_norm",
    "e2_m_dist_norm",
    "midpoint_m_dist_norm",

    # Ellipse-normalized coordinates (relative to ellipse)
    "left_eye_x1_norm",
    "left_eye_x2_norm",
    "right_eye_x1_norm",
    "right_eye_x2_norm",
    "mouth_x1_norm",
    "mouth_x2_norm",

    # Distance from ellipse center (normalized)
    "left_eye_center_dist_norm",
    "right_eye_center_dist_norm",
    "mouth_center_dist_norm",

    # Angles (relative geometry)
    "angle_eyes_mouth",
    "angle_to_major_axis",

    # Ellipse shape (relative)
    "ellipse_ab_ratio",

    # Ellipse orientation
    "major_axis_x",
    "major_axis_y",
    "minor_axis_x",
    "minor_axis_y",

    # Map response values
    "left_eye_value",
    "right_eye_value",
    "mouth_value",
]


def main():
    csv_path = "triplet_dataset.csv"

    df = pd.read_csv(csv_path)

    # 只保留你已經手動標好的資料
    # label = 1 正確 triplet
    # label = 0 錯誤 triplet
    # label = -1 尚未標記，不拿來訓練
    df = df[df["label"].isin([0, 1])]

    print("Total labeled samples:", len(df))
    print("Label counts:")
    print(df["label"].value_counts())

    if len(df) == 0:
        raise ValueError("沒有任何已標記資料，請確認 triplet_dataset.csv 的 label 有 0 或 1。")

    if df["label"].nunique() < 2:
        raise ValueError("label 只有一類，至少需要 label=0 和 label=1 都存在。")

    missing_cols = [c for c in FEATURE_NAMES if c not in df.columns]
    if missing_cols:
        raise ValueError(f"缺少這些 feature 欄位: {missing_cols}")

    X = df[FEATURE_NAMES]
    y = df["label"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        class_weight="balanced",
        random_state=42
    )

    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    print("\nAccuracy:", accuracy_score(y_test, pred))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, pred))

    print("\nClassification Report:")
    print(classification_report(y_test, pred))

    joblib.dump(model, "triplet_scorer_rf.pkl")
    joblib.dump(FEATURE_NAMES, "triplet_feature_names.pkl")

    print("\nModel saved to triplet_scorer_rf.pkl")
    print("Feature names saved to triplet_feature_names.pkl")


if __name__ == "__main__":
    main()