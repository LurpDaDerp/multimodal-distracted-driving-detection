#offline feature extraction using mediapipe face, hand, and pose detection models (save to json)

#pip install mediapipe tqdm opencv-python pandas


import os
import json
import cv2
import shutil
import urllib.request
from pathlib import Path
from tqdm.auto import tqdm
import pandas as pd

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


DATA_ROOT = Path(r"../Datasets/statefarm") 
IMG_DIR = DATA_ROOT / "imgs" / "train"
CSV_PATH = DATA_ROOT / "driver_imgs_list.csv"

if not CSV_PATH.exists():
    raise FileNotFoundError(f"Missing {CSV_PATH}. Check path.")


#get models
MODEL_DIR = Path("./models")
MODEL_DIR.mkdir(exist_ok=True)

models_to_fetch = {
    "pose_landmarker_lite.task": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
    
    "hand_landmarker.task": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
    
    "face_landmarker.task": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
}

print("Downloading models")
for filename, url in models_to_fetch.items():
    path = MODEL_DIR / filename
    if not path.exists():
        print(f"Downloading {filename}...")
        with urllib.request.urlopen(url) as response, open(path, 'wb') as out_file:
            shutil.copyfileobj(response, out_file)
        print("Done.")
    else:
        print(f"Found {filename} (Local)")


#local detectors
BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode

#pose 
pose_options = vision.PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=str(MODEL_DIR / "pose_landmarker_lite.task")),
    running_mode=VisionRunningMode.IMAGE,
    num_poses=1,
    min_pose_detection_confidence=0.5
)
pose_detector = vision.PoseLandmarker.create_from_options(pose_options)

#hands
hand_options = vision.HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=str(MODEL_DIR / "hand_landmarker.task")),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=2, 
    min_hand_detection_confidence=0.3  
)
hand_detector = vision.HandLandmarker.create_from_options(hand_options)

#face
face_options = vision.FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=str(MODEL_DIR / "face_landmarker.task")),
    running_mode=VisionRunningMode.IMAGE,
    num_faces=1,
    min_face_detection_confidence=0.5
)
face_detector = vision.FaceLandmarker.create_from_options(face_options)

print("All Detectors Loaded Locally.")


#extraction function
def extract_features(img_path):
    try:
        mp_image = mp.Image.create_from_file(str(img_path))
    except Exception as e:
        return None

    #pose (33 points * 3 = 99 floats) 
    pose_data = [0.0] * 99
    pose_res = pose_detector.detect(mp_image)
    if pose_res.pose_landmarks:
        flat = []
        for lm in pose_res.pose_landmarks[0]:
            flat.extend([lm.x, lm.y, lm.z])
        pose_data = flat

    #hands (2 hands * 21 points * 3 = 126 floats) 
    hand_data = [0.0] * 126
    hand_res = hand_detector.detect(mp_image)
    if hand_res.hand_landmarks:
        flat = []
        for hand_lms in hand_res.hand_landmarks:
            for lm in hand_lms:
                flat.extend([lm.x, lm.y, lm.z])
        
        #truncate to 126 floats
        count = min(len(flat), 126)
        hand_data[:count] = flat[:count]

    #face (478 points * 3 = 1434 floats) 
    face_data = [0.0] * 1434
    face_res = face_detector.detect(mp_image)
    if face_res.face_landmarks:
        flat = []
        for lm in face_res.face_landmarks[0]:
            flat.extend([lm.x, lm.y, lm.z])
        face_data = flat

    return {
        "pose": pose_data,
        "hand": hand_data,
        "face": face_data
    }


#load dataset csv list
df = pd.read_csv(CSV_PATH)
df["path"] = df.apply(lambda r: IMG_DIR / r["classname"] / r["img"], axis=1)

print(f"Total Images: {len(df)}")


#preporcess feature extraction
OUTPUT_JSON = DATA_ROOT / "driver_landmarks.json"
cache = {}

if OUTPUT_JSON.exists():
    print("Resuming from existing JSON")
    with open(OUTPUT_JSON, "r") as f:
        cache = json.load(f)

print(f"Starting Processing ({len(cache)} done)")

for idx, row in tqdm(df.iterrows(), total=len(df)):
    img_name = row["img"]
    
    if img_name in cache: continue
    
    if not row["path"].exists(): continue
        
    feats = extract_features(row["path"])
    
    if feats:
        cache[img_name] = feats
        
    #save every 500 images
    if idx % 500 == 0:
        with open(OUTPUT_JSON, "w") as f:
            json.dump(cache, f)

#final save
with open(OUTPUT_JSON, "w") as f:
    json.dump(cache, f)

print(f"Finished. Driver face, hands, pose landmarks saved to: {OUTPUT_JSON}")