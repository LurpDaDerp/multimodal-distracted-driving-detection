import json
import random
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt


# =========================================================
# CONFIG
# =========================================================
DATA_ROOT = Path(r"../Datasets/100-driver")
TIME_OF_DAY = "Day"   # "Day" or "Night"
CAMERAS = ["Cam1", "Cam2", "Cam3"]
IMAGES_PER_CAMERA = 2
# SEED = 42

# random.seed(SEED)

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# =========================================================
# HELPERS
# =========================================================
def load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing JSON: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def json_path_for_camera(data_root: Path, time_of_day: str, camera: str) -> Path:
    return data_root / f"driver_landmarks_{time_of_day.lower()}_{camera.lower()}.json"


def collect_existing_keys(cam_root: Path, landmark_dict: dict):
    """
    Returns a list of split-relative keys that both:
    - exist in the landmark json
    - exist as actual image files under cam_root
    """
    valid = []
    for rel_key in landmark_dict.keys():
        img_path = cam_root / rel_key
        if img_path.exists() and img_path.suffix.lower() in SUPPORTED_EXTS:
            valid.append(rel_key)
    return valid


def reshape_landmarks(flat, n_points):
    """
    Converts [x1,y1,z1,x2,y2,z2,...] into Nx3 array.
    Pads/truncates if needed.
    """
    expected_len = n_points * 3
    flat = list(flat) if flat is not None else []
    if len(flat) < expected_len:
        flat = flat + [0.0] * (expected_len - len(flat))
    elif len(flat) > expected_len:
        flat = flat[:expected_len]
    return np.array(flat, dtype=np.float32).reshape(n_points, 3)


def draw_landmark_points(image_bgr, pts_xyz, color, radius=1, min_conf_style=False):
    """
    Draws normalized x,y landmark points onto image.
    Ignores points that are exactly (0,0,0).
    """
    out = image_bgr.copy()
    h, w = out.shape[:2]

    for x, y, z in pts_xyz:
        if x == 0.0 and y == 0.0 and z == 0.0:
            continue

        px = int(x * w)
        py = int(y * h)

        if 0 <= px < w and 0 <= py < h:
            cv2.circle(out, (px, py), radius, color, -1)

    return out


def annotate_from_features(image_bgr, feats):
    """
    Draw face, hand, and pose points with different colors.
    """
    out = image_bgr.copy()

    pose = reshape_landmarks(feats.get("pose", []), 33)
    hand = reshape_landmarks(feats.get("hand", []), 42)   # 2 hands * 21
    face = reshape_landmarks(feats.get("face", []), 478)

    # BGR colors
    out = draw_landmark_points(out, face, color=(0, 255, 255), radius=1)   # yellow
    out = draw_landmark_points(out, hand, color=(255, 0, 0), radius=2)     # blue
    out = draw_landmark_points(out, pose, color=(0, 255, 0), radius=3)     # green

    return out


def sample_images_for_camera(data_root: Path, time_of_day: str, camera: str, n: int):
    cam_root = data_root / time_of_day / camera
    landmark_json_path = json_path_for_camera(data_root, time_of_day, camera)
    landmark_dict = load_json(landmark_json_path)

    valid_keys = collect_existing_keys(cam_root, landmark_dict)
    if len(valid_keys) == 0:
        raise RuntimeError(f"No valid images found for {time_of_day}/{camera} using {landmark_json_path}")

    chosen = random.sample(valid_keys, k=min(n, len(valid_keys)))

    samples = []
    for rel_key in chosen:
        img_path = cam_root / rel_key
        feats = landmark_dict[rel_key]
        samples.append({
            "camera": camera,
            "rel_key": rel_key,
            "img_path": img_path,
            "feats": feats,
        })

    return samples


# =========================================================
# MAIN
# =========================================================
all_samples = []

for camera in CAMERAS:
    samples = sample_images_for_camera(DATA_ROOT, TIME_OF_DAY, camera, IMAGES_PER_CAMERA)
    all_samples.extend(samples)

print(f"Collected {len(all_samples)} samples total.")
for s in all_samples:
    print(f"{s['camera']}: {s['rel_key']}")

# Display in grid: rows = cameras, cols = images per camera
rows = len(CAMERAS)
cols = IMAGES_PER_CAMERA

fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
if rows == 1 and cols == 1:
    axes = np.array([[axes]])
elif rows == 1:
    axes = np.array([axes])
elif cols == 1:
    axes = np.array([[ax] for ax in axes])

sample_idx = 0
for r, camera in enumerate(CAMERAS):
    camera_samples = [s for s in all_samples if s["camera"] == camera]

    for c in range(cols):
        ax = axes[r, c]

        if c >= len(camera_samples):
            ax.axis("off")
            continue

        sample = camera_samples[c]
        img_bgr = cv2.imread(str(sample["img_path"]))

        if img_bgr is None:
            ax.set_title(f"{camera}\nFailed to load")
            ax.axis("off")
            continue

        annotated = annotate_from_features(img_bgr, sample["feats"])
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

        ax.imshow(annotated_rgb)
        ax.axis("off")
        ax.set_title(f"{camera}\n{sample['rel_key']}", fontsize=10)

plt.tight_layout()
out_path = "landmark_preview.png"
plt.savefig(out_path, dpi=200, bbox_inches="tight")
print(f"Saved preview to {out_path}")