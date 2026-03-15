# offline feature extraction using MediaPipe face, hand, and pose models
# for the 100-driver dataset directory structure:
#
# 100-driver/
#   Day/
#     Cam1/
#       22 class folders...
#     Cam2/
#     Cam3/
#     Cam4/
#   Night/
#     Cam1/
#     Cam2/
#     Cam3/
#     Cam4/
#
# pip install mediapipe tqdm

import json
import shutil
import urllib.request
from pathlib import Path
from tqdm.auto import tqdm

import mediapipe as mp
from mediapipe.tasks.python import vision


# =========================================================
# CONFIG
# =========================================================
DATA_ROOT = Path(r"../Datasets/100-driver")

# Set to None to auto-discover all existing Day/Cam and Night/Cam folders.
# Or set something like:
# ONLY_SUBSETS = [("Day", "Cam1")]
ONLY_SUBSETS = None

MODEL_DIR = Path("./models")
MODEL_DIR.mkdir(exist_ok=True)

MASTER_JSON = DATA_ROOT / "driver_landmarks_master.json"
SAVE_EVERY = 500

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Per-split JSON naming:
# Day/Cam1  -> driver_landmarks_day_cam1.json
# Night/Cam3 -> driver_landmarks_night_cam3.json
def per_split_json_path(time_of_day: str, camera: str) -> Path:
    return DATA_ROOT / f"driver_landmarks_{time_of_day.lower()}_{camera.lower()}.json"


# =========================================================
# MODEL DOWNLOAD
# =========================================================
models_to_fetch = {
    "pose_landmarker_lite.task":
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
    "hand_landmarker.task":
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
    "face_landmarker.task":
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
}

print("Checking MediaPipe models...")
for filename, url in models_to_fetch.items():
    path = MODEL_DIR / filename
    if not path.exists():
        print(f"Downloading {filename}...")
        with urllib.request.urlopen(url) as response, open(path, "wb") as out_file:
            shutil.copyfileobj(response, out_file)
        print(f"Saved to {path}")
    else:
        print(f"Found {filename} locally.")


# =========================================================
# DETECTOR SETUP
# =========================================================
BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode

pose_options = vision.PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=str(MODEL_DIR / "pose_landmarker_lite.task")),
    running_mode=VisionRunningMode.IMAGE,
    num_poses=1,
    min_pose_detection_confidence=0.5,
)

hand_options = vision.HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=str(MODEL_DIR / "hand_landmarker.task")),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=2,
    min_hand_detection_confidence=0.3,
)

face_options = vision.FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=str(MODEL_DIR / "face_landmarker.task")),
    running_mode=VisionRunningMode.IMAGE,
    num_faces=1,
    min_face_detection_confidence=0.5,
)

pose_detector = vision.PoseLandmarker.create_from_options(pose_options)
hand_detector = vision.HandLandmarker.create_from_options(hand_options)
face_detector = vision.FaceLandmarker.create_from_options(face_options)

print("All detectors loaded.")


# =========================================================
# HELPERS
# =========================================================
def load_json_if_exists(path: Path):
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_json(path: Path, data: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)

def discover_subsets(data_root: Path):
    """Finds existing Day/CamX and Night/CamX folders."""
    subsets = []
    for tod_dir in sorted(data_root.iterdir()):
        if not tod_dir.is_dir():
            continue
        if tod_dir.name not in {"Day", "Night"}:
            continue

        for cam_dir in sorted(tod_dir.iterdir()):
            if not cam_dir.is_dir():
                continue
            if not cam_dir.name.startswith("Cam"):
                continue

            subsets.append((tod_dir.name, cam_dir.name))
    return subsets

def list_images_in_subset(split_root: Path):
    """
    Returns a list of image metadata dicts for all images under one subset root.
    Example split_root: DATA_ROOT / 'Day' / 'Cam1'
    """
    records = []

    if not split_root.exists():
        return records

    for class_dir in sorted(split_root.iterdir()):
        if not class_dir.is_dir():
            continue

        for img_path in sorted(class_dir.rglob("*")):
            if not img_path.is_file():
                continue
            if img_path.suffix.lower() not in SUPPORTED_EXTS:
                continue

            split_rel = img_path.relative_to(split_root).as_posix()   # e.g. C1_Drive_Safe/xxx.jpg
            full_rel = img_path.relative_to(DATA_ROOT).as_posix()     # e.g. Day/Cam1/C1_Drive_Safe/xxx.jpg

            records.append({
                "img_path": img_path,
                "split_rel": split_rel,
                "full_rel": full_rel,
            })

    return records

def fix_length(values, target_len):
    """Pads/truncates a flat feature list to the exact target length."""
    if values is None:
        values = []
    values = list(values)

    if len(values) < target_len:
        values.extend([0.0] * (target_len - len(values)))
    elif len(values) > target_len:
        values = values[:target_len]

    return values

def extract_features(img_path: Path):
    """
    Extracts:
      pose: 33 * 3 = 99 floats
      hand: up to 2 * 21 * 3 = 126 floats
      face: 478 * 3 = 1434 floats
    """
    try:
        mp_image = mp.Image.create_from_file(str(img_path))
    except Exception as e:
        print(f"Failed to read image: {img_path} | {e}")
        return None

    # Pose
    pose_data = [0.0] * 99
    try:
        pose_res = pose_detector.detect(mp_image)
        if pose_res.pose_landmarks:
            flat = []
            for lm in pose_res.pose_landmarks[0]:
                flat.extend([float(lm.x), float(lm.y), float(lm.z)])
            pose_data = fix_length(flat, 99)
    except Exception as e:
        print(f"Pose detect failed: {img_path} | {e}")

    # Hands
    hand_data = [0.0] * 126
    try:
        hand_res = hand_detector.detect(mp_image)
        if hand_res.hand_landmarks:
            flat = []
            for hand_lms in hand_res.hand_landmarks:
                for lm in hand_lms:
                    flat.extend([float(lm.x), float(lm.y), float(lm.z)])
            hand_data = fix_length(flat, 126)
    except Exception as e:
        print(f"Hand detect failed: {img_path} | {e}")

    # Face
    face_data = [0.0] * 1434
    try:
        face_res = face_detector.detect(mp_image)
        if face_res.face_landmarks:
            flat = []
            for lm in face_res.face_landmarks[0]:
                flat.extend([float(lm.x), float(lm.y), float(lm.z)])
            face_data = fix_length(flat, 1434)
    except Exception as e:
        print(f"Face detect failed: {img_path} | {e}")

    return {
        "pose": pose_data,
        "hand": hand_data,
        "face": face_data,
    }


# =========================================================
# MAIN
# =========================================================
if ONLY_SUBSETS is None:
    subsets = discover_subsets(DATA_ROOT)
else:
    subsets = ONLY_SUBSETS

if not subsets:
    raise RuntimeError(f"No Day/Cam or Night/Cam subsets found under {DATA_ROOT}")

print("\nSubsets to process:")
for tod, cam in subsets:
    print(f" - {tod}/{cam}")

master_cache = load_json_if_exists(MASTER_JSON)
print(f"\nLoaded master cache with {len(master_cache)} entries from {MASTER_JSON}" if MASTER_JSON.exists()
      else "\nNo existing master cache found. Starting fresh.")

total_processed = 0
total_extracted = 0
total_skipped_existing = 0
total_failed = 0

for time_of_day, camera in subsets:
    split_root = DATA_ROOT / time_of_day / camera
    split_json = per_split_json_path(time_of_day, camera)
    split_cache = load_json_if_exists(split_json)

    print(f"\n{'=' * 70}")
    print(f"Processing subset: {time_of_day}/{camera}")
    print(f"Image root: {split_root}")
    print(f"Per-split JSON: {split_json}")
    print(f"Existing per-split entries: {len(split_cache)}")

    records = list_images_in_subset(split_root)
    print(f"Images found: {len(records)}")

    for i, rec in enumerate(tqdm(records, desc=f"{time_of_day}/{camera}"), start=1):
        img_path = rec["img_path"]
        split_rel = rec["split_rel"]   # e.g. C1_Drive_Safe/abc.jpg
        full_rel = rec["full_rel"]     # e.g. Day/Cam1/C1_Drive_Safe/abc.jpg

        total_processed += 1

        # Reuse existing features if already present in either cache
        existing_feats = None
        if full_rel in master_cache:
            existing_feats = master_cache[full_rel]
        elif split_rel in split_cache:
            existing_feats = split_cache[split_rel]

        if existing_feats is not None:
            master_cache.setdefault(full_rel, existing_feats)
            split_cache.setdefault(split_rel, existing_feats)
            total_skipped_existing += 1
            continue

        feats = extract_features(img_path)
        if feats is None:
            total_failed += 1
            continue

        master_cache[full_rel] = feats
        split_cache[split_rel] = feats
        total_extracted += 1

        if total_extracted % SAVE_EVERY == 0:
            save_json(MASTER_JSON, master_cache)
            save_json(split_json, split_cache)

    # Save after each subset
    save_json(split_json, split_cache)
    save_json(MASTER_JSON, master_cache)

    print(f"Saved per-split JSON: {split_json} ({len(split_cache)} entries)")

# Final save
save_json(MASTER_JSON, master_cache)

print(f"\n{'=' * 70}")
print("Finished.")
print(f"Master JSON: {MASTER_JSON}")
print(f"Master entries: {len(master_cache)}")
print(f"Total images seen: {total_processed}")
print(f"Newly extracted: {total_extracted}")
print(f"Skipped from existing cache: {total_skipped_existing}")
print(f"Failed reads/extractions: {total_failed}")

print("\nPer-split landmark JSONs created in:")
print(DATA_ROOT)
print("Examples:")
print(" - driver_landmarks_day_cam1.json")
print(" - driver_landmarks_day_cam2.json")
print(" - driver_landmarks_night_cam1.json")
print(" - driver_landmarks_night_cam4.json")